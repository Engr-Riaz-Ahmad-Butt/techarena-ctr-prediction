"""
V21 push experiments against frozen V20 first-event validation cohorts.

Usage (one at a time):
  python v21_experiments.py --exp sequence
  python v21_experiments.py --exp cooccur
  python v21_experiments.py --exp recency
  python v21_experiments.py --exp blend
  python v21_experiments.py --exp finalize   # only if blend beat V20 on both days

Does NOT overwrite submission.csv. Writes submission_v21.csv only from finalize.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from config import ARTIFACT_DIR as ROOT
from config import EVALUATION_CSV

SEED = 42
DAY = 86400
V20 = {6: 0.8215282579633503, 5: 0.814599936313616}
LOG = Path("results_log.md")
np.random.seed(SEED)
torch.manual_seed(SEED)

train = np.load(ROOT / "train.npy", mmap_mode="r")
hist = np.load(ROOT / "train_hist.npy", mmap_mode="r")
htimes = np.load(ROOT / "train_times.npy", mmap_mode="r")
evaluation = np.load(ROOT / "evaluation.npy", mmap_mode="r")
ehist = np.load(ROOT / "evaluation_hist.npy", mmap_mode="r")
etimes = np.load(ROOT / "evaluation_times.npy", mmap_mode="r")
ids = json.loads((ROOT / "ids.json").read_text())
NU, NI = len(ids["users"]), len(ids["items"])
del ids
START = int(train[:, 2].min())


def append_log(step, change, before, after, runtime, kept, notes=""):
    line = (
        f"| {step} | {change} | {before:.6f} → {after:.6f} | "
        f"{runtime:.0f}s | {'YES' if kept else 'no'} | {notes} |\n"
    )
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    if "| Step |" not in text:
        text = (
            "# TechArena CTR — results log\n\n"
            "| Step | Change | Val AUC (before → after) | Runtime | Kept? | Notes |\n"
            "|------|--------|--------------------------|---------|-------|-------|\n"
        )
    lines = text.splitlines(True)
    insert_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("|"):
            insert_at = i + 1
            break
    lines.insert(insert_at, line)
    LOG.write_text("".join(lines), encoding="utf-8")


def first_event_block(day: int):
    later = np.asarray(
        train[(train[:, 2] >= START + day * DAY) & (train[:, 2] < START + (day + 1) * DAY)]
    )
    events = later[::10]
    _, first = np.unique(events[:, 0], return_index=True)
    return later[(first[:, None] * 10 + np.arange(10)).ravel()]


def load_v20(day: int):
    ref = np.load(ROOT / f"imp19_tree_day{day}.npz")
    val = first_event_block(day)
    assert np.array_equal(val[:, 6], ref["events"]) and np.array_equal(val[:, 5], ref["y"])
    auc = float(roc_auc_score(ref["y"], ref["pred"]))
    assert abs(auc - V20[day]) < 1e-12, (auc, V20[day])
    return val, ref["y"], ref["events"], ref["pred"].astype(np.float64)


def fit_window(day: int):
    """Labeled impressions in [day-2, day) — same style as early pipelines."""
    return np.asarray(
        train[(train[:, 2] >= START + (day - 2) * DAY) & (train[:, 2] < START + day * DAY)]
    )


def rank_blend(base, raw, weight):
    br = rankdata(base, method="average") / (len(base) + 1)
    rr = rankdata(raw, method="average") / (len(raw) + 1)
    return (1 - weight) * br + weight * rr


def best_rank_blend(y, base, raw, weights=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    base_auc = float(roc_auc_score(y, base))
    best = {"weight": 0.0, "auc": base_auc, "raw_auc": float(roc_auc_score(y, raw)), "pred": base}
    for w in weights:
        pred = rank_blend(base, raw, w)
        auc = float(roc_auc_score(y, pred))
        if auc > best["auc"]:
            best = {"weight": w, "auc": auc, "raw_auc": best["raw_auc"], "pred": pred}
    best["delta"] = best["auc"] - base_auc
    best["base_auc"] = base_auc
    return best


def audit_leakage(name, day, cutoff_used, val):
    """Basic time-safety asserts."""
    assert int(val[:, 2].min()) >= cutoff_used
    h = np.asarray(hist[val[:, 6]])
    t = np.asarray(htimes[val[:, 6]])
    valid = h >= 0
    assert np.all((t < val[:, 2, None]) | ~valid)
    print(f"Leakage audit OK for {name} day{day}: histories before event; stats cut={cutoff_used}", flush=True)


# ---------------------------------------------------------------------------
# Shared: time-safe item embeddings from histories+clicks before cutoff
# ---------------------------------------------------------------------------
def item_embeddings(cutoff, dim=32):
    cache = ROOT / f"v21_emb_cut{cutoff}_d{dim}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["emb"], float(z["g"])
    a = np.asarray(train[train[:, 2] < cutoff])
    y = a[:, 5].astype(np.float64)
    g = float(y.mean())
    events = a[::10]
    _, first = np.unique(events[:, 0], return_index=True)
    fe = events[first]
    h = np.asarray(hist[fe[:, 6]])
    hu = np.repeat(fe[:, 0], 20)
    hi = h.ravel()
    valid = hi >= 0
    clicked = a[y == 1]
    rows = np.concatenate([hu[valid], clicked[:, 0]])
    cols = np.concatenate([hi[valid], clicked[:, 1]])
    matrix = coo_matrix((np.ones(len(rows), dtype="float32"), (rows, cols)), shape=(NU, NI)).tocsr()
    matrix.data = np.log1p(matrix.data)
    idf = np.log1p(NU / (1 + np.asarray((matrix > 0).sum(axis=0)).ravel()))
    weighted = normalize(matrix.multiply(idf).tocsr(), norm="l2")
    svd = TruncatedSVD(n_components=dim, n_iter=4, random_state=SEED)
    svd.fit(weighted)
    emb = normalize(svd.components_.T.copy()).astype("float32")
    np.savez_compressed(cache, emb=emb, g=np.array(g))
    print(f"Embeddings ready cut={cutoff} dim={dim}", flush=True)
    return emb, g


# ---------------------------------------------------------------------------
# Exp 1: GRU sequence model
# ---------------------------------------------------------------------------
class HistoryGRUScorer(nn.Module):
    def __init__(self, n_items, dim=32, hidden=64):
        super().__init__()
        self.pad = n_items
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.cand = nn.Embedding(n_items, dim)
        self.cand_proj = nn.Linear(dim, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, hist_ids, lengths, cand_ids):
        x = self.emb(hist_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h.squeeze(0)
        c = self.cand_proj(self.cand(cand_ids))
        dot = (h * c).sum(-1, keepdim=True)
        logits = self.scorer(torch.cat([h, c, dot], dim=-1)).squeeze(-1)
        return logits


def _pack_histories(rows, histories, pad_id):
    h = np.asarray(histories[rows[:, 6]])
    valid = h >= 0
    lengths = valid.sum(1).astype(np.int64)
    lengths = np.maximum(lengths, 1)
    # left-pad oldest→newest so last timestep is most recent
    packed = np.full((len(rows), 20), pad_id, dtype=np.int64)
    for i in range(len(rows)):
        L = int(valid[i].sum())
        if L:
            packed[i, 20 - L :] = h[i, :L]
    return packed, lengths


def _train_sequence_model(fit_rows, val_rows, y_val, emb_np, epochs=5):
    """Within-event softmax GRU: histories shared across the 10 candidates."""
    rng = np.random.default_rng(SEED)
    pad = NI
    model = HistoryGRUScorer(NI, dim=32, hidden=64)
    with torch.no_grad():
        model.cand.weight.copy_(torch.from_numpy(emb_np))
        model.emb.weight[:-1].copy_(torch.from_numpy(emb_np))
    opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)

    n_events = len(fit_rows) // 10
    take = min(n_events, 100000)
    chosen = np.sort(rng.choice(n_events, size=take, replace=False))

    def event_batch(rows, event_ids, bs=512, shuffle=True):
        eids = event_ids.copy()
        if shuffle:
            rng.shuffle(eids)
        for i in range(0, len(eids), bs):
            ev = eids[i : i + bs]
            # each event: 10 contiguous rows
            block = rows[(ev[:, None] * 10 + np.arange(10)).ravel()].reshape(len(ev), 10, -1)
            # histories identical within event — take row 0
            head = block[:, 0, :]
            hp, lens = _pack_histories(head, hist, pad)
            cands = block[:, :, 1].astype(np.int64)
            labels = block[:, :, 5].astype(np.float32)
            yield (
                torch.from_numpy(hp),
                torch.from_numpy(lens),
                torch.from_numpy(cands),
                torch.from_numpy(labels),
            )

    def predict(rows):
        model.eval()
        out = np.empty(len(rows), dtype=np.float32)
        n_ev = len(rows) // 10
        with torch.no_grad():
            for start in range(0, n_ev, 512):
                ev = np.arange(start, min(start + 512, n_ev))
                block = rows[(ev[:, None] * 10 + np.arange(10)).ravel()].reshape(len(ev), 10, -1)
                head = block[:, 0, :]
                hp, lens = _pack_histories(head, hist, pad)
                cands = block[:, :, 1].astype(np.int64)
                hp_t = torch.from_numpy(hp)
                lens_t = torch.from_numpy(lens)
                # encode history once, score 10 cands
                x = model.emb(hp_t)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lens_t.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
                )
                _, h = model.gru(packed)
                h = h.squeeze(0)
                logits = []
                for j in range(10):
                    c = model.cand_proj(model.cand(torch.from_numpy(cands[:, j])))
                    dot = (h * c).sum(-1, keepdim=True)
                    logits.append(model.scorer(torch.cat([h, c, dot], dim=-1)).squeeze(-1))
                logits = torch.stack(logits, dim=1)
                prob = torch.softmax(logits, dim=1).numpy()
                out[(ev[:, None] * 10 + np.arange(10)).ravel()] = prob.ravel()
        return out

    best_state, best_auc, best_raw = None, -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for hp, lens, cands, labels in event_batch(fit_rows, chosen, bs=256):
            opt.zero_grad()
            x = model.emb(hp)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, h = model.gru(packed)
            h = h.squeeze(0)
            logits = []
            for j in range(10):
                c = model.cand_proj(model.cand(cands[:, j]))
                dot = (h * c).sum(-1, keepdim=True)
                logits.append(model.scorer(torch.cat([h, c, dot], dim=-1)).squeeze(-1))
            logits = torch.stack(logits, dim=1)
            # events with a click: softmax CE; no-click events: soft push-down via BCE on probs
            has_click = labels.sum(1) > 0
            loss = logits.new_zeros(())
            if has_click.any():
                target = labels[has_click].argmax(1)
                loss = loss + nn.functional.cross_entropy(logits[has_click], target)
            if (~has_click).any():
                # discourage confident clicks when none occurred
                loss = loss + 0.1 * torch.softmax(logits[~has_click], 1).pow(2).mean()
            if float(loss) == 0.0:
                continue
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(labels)
            n += len(labels)
        raw = predict(val_rows)
        auc = float(roc_auc_score(y_val, raw))
        print(f"Sequence-softmax epoch {epoch} loss={total/max(n,1):.5f} raw_auc={auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_raw = raw
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_raw, best_auc


def run_sequence(day=6):
    t0 = time.time()
    val, y, events, base = load_v20(day)
    emb_np, _ = item_embeddings(START + (day - 2) * DAY, dim=32)
    fit = fit_window(day)
    print(f"Sequence-softmax train events<=100k val={len(val)}", flush=True)
    _, raw, _ = _train_sequence_model(fit, val, y, emb_np, epochs=5)
    np.savez_compressed(ROOT / f"v21_sequence_raw_day{day}.npz", y=y, events=events, raw=raw)
    blend = best_rank_blend(y, base, raw)
    audit_leakage("sequence", day, START + (day - 2) * DAY, val)

    val5, y5, ev5, base5 = load_v20(5)
    emb5, _ = item_embeddings(START + (5 - 2) * DAY, dim=32)
    fit5 = fit_window(5)
    _, raw5, _ = _train_sequence_model(fit5, val5, y5, emb5, epochs=5)
    np.savez_compressed(ROOT / "v21_sequence_raw_day5.npz", y=y5, events=ev5, raw=raw5)
    blend5 = best_rank_blend(y5, base5, raw5, weights=[blend["weight"]] if blend["weight"] > 0 else [0.05, 0.1, 0.2, 0.3])
    # if day6 selected weight 0, still report best day5 frozen-grid under day6 best positive weight attempt
    if blend["weight"] == 0:
        blend5 = best_rank_blend(y5, base5, raw5)

    report = {
        "exp": "sequence",
        "day6": {
            "raw_auc": blend["raw_auc"],
            "blend_auc": blend["auc"],
            "weight": blend["weight"],
            "base_auc": blend["base_auc"],
            "delta": blend["delta"],
        },
        "day5": {
            "raw_auc": float(roc_auc_score(y5, raw5)),
            "blend_auc": blend5["auc"],
            "weight": blend5["weight"],
            "base_auc": blend5["base_auc"],
            "delta": blend5["delta"],
        },
    }
    kept = blend["auc"] > blend["base_auc"] + 1e-6 and blend5["auc"] > blend5["base_auc"] + 1e-6
    if blend["delta"] > 0.01 or blend5["delta"] > 0.01:
        print("WARNING: gain >0.01 — leakage audit already run on histories/cutoff", flush=True)
    (ROOT / "v21_sequence_report.json").write_text(json.dumps(report, indent=2))
    if kept:
        np.savez_compressed(
            ROOT / "v21_sequence_day6.npz", y=y, events=events, pred=blend["pred"], raw=raw
        )
        np.savez_compressed(
            ROOT / "v21_sequence_day5.npz", y=y5, events=ev5, pred=blend5["pred"], raw=raw5
        )
    runtime = time.time() - t0
    append_log(
        "V21-1b",
        "GRU + within-event softmax over 20-click history (+ rank-blend vs V20)",
        blend["base_auc"],
        blend["auc"],
        runtime,
        kept,
        f"raw={blend['raw_auc']:.6f} w={blend['weight']} day5_blend={blend5['auc']:.6f}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


# ---------------------------------------------------------------------------
# Exp 2: time-safe item-item co-occurrence from histories
# ---------------------------------------------------------------------------
def build_cooccur(cutoff):
    """Undirected co-occurrence from first-per-user histories + hist→click pairs (time-safe)."""
    cache = ROOT / f"v21_cooccur_cut{cutoff}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["keys"], z["counts"], z["item_count"]
    a = np.asarray(train[train[:, 2] < cutoff])
    events = a[::10]
    _, first = np.unique(events[:, 0], return_index=True)
    fe = events[first]
    h = np.asarray(hist[fe[:, 6]])
    item_count = np.zeros(NI, dtype=np.int64)
    # count item presence in first histories
    valid = h >= 0
    item_count += np.bincount(h[valid], minlength=NI)
    # within-history undirected pairs via expanding windows (vectorized per lag)
    rows_idx = []
    cols_idx = []
    for j in range(20):
        for k in range(j + 1, 20):
            m = valid[:, j] & valid[:, k]
            if not m.any():
                continue
            left = h[m, j]
            right = h[m, k]
            lo = np.minimum(left, right)
            hi = np.maximum(left, right)
            # drop self-pairs
            keep = lo != hi
            rows_idx.append(lo[keep])
            cols_idx.append(hi[keep])
    # history -> labeled click pairs
    clicked = a[a[:, 5] == 1]
    hh = np.asarray(hist[clicked[:, 6]])
    hv = hh >= 0
    for j in range(20):
        m = hv[:, j]
        if not m.any():
            continue
        left = hh[m, j]
        right = clicked[m, 1]
        lo = np.minimum(left, right)
        hi = np.maximum(left, right)
        keep = lo != hi
        rows_idx.append(lo[keep])
        cols_idx.append(hi[keep])
    r = np.concatenate(rows_idx).astype(np.int64)
    c = np.concatenate(cols_idx).astype(np.int64)
    co = coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(NI, NI))
    co.sum_duplicates()
    keys = co.row.astype(np.int64) * NI + co.col.astype(np.int64)
    counts = co.data.astype(np.float32)
    order = np.argsort(keys)
    keys, counts = keys[order], counts[order]
    np.savez_compressed(cache, keys=keys, counts=counts, item_count=item_count)
    print(f"Co-occur pairs={len(keys)} cut={cutoff}", flush=True)
    return keys, counts, item_count


def lookup_pair(keys, counts, a, b):
    """Vectorized pair lookup for undirected (min,max) keys."""
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    k = lo * NI + hi
    idx = np.searchsorted(keys, k)
    idx = np.clip(idx, 0, len(keys) - 1)
    found = keys[idx] == k
    out = np.zeros(len(k), dtype=np.float32)
    out[found] = counts[idx[found]]
    return out


def cooccur_scores(rows, histories, keys, counts, item_count, n_users_hist):
    """Per-row: mean/max/sum PMI-ish between candidate and history items."""
    g = max(float(n_users_hist), 1.0)
    out = np.empty((len(rows), 6), dtype=np.float32)
    for off in range(0, len(rows), 20000):
        b = rows[off : off + 20000]
        h = np.asarray(histories[b[:, 6]])
        valid = h >= 0
        cand = b[:, 1]
        # for each of 20 hist positions
        sums = np.zeros(len(b), dtype=np.float64)
        maxes = np.full(len(b), -1e9)
        pm_sums = np.zeros(len(b), dtype=np.float64)
        hit = np.zeros(len(b), dtype=np.float64)
        for j in range(20):
            src = h[:, j]
            m = valid[:, j]
            if not m.any():
                continue
            co = np.zeros(len(b), dtype=np.float32)
            co[m] = lookup_pair(keys, counts, src[m], cand[m])
            # PMI ≈ log(co * N / (cnt_i * cnt_j))
            ci = np.maximum(item_count[np.maximum(src, 0)], 1)
            cj = np.maximum(item_count[cand], 1)
            pmi = np.zeros(len(b), dtype=np.float64)
            pmi[m] = np.log1p(co[m]) - 0.5 * (np.log1p(ci[m]) + np.log1p(cj[m])) + np.log1p(g)
            sums += co
            pm_sums += pmi * m
            maxes = np.maximum(maxes, np.where(m, pmi, -1e9))
            hit += (co > 0) & m
        n = np.maximum(valid.sum(1), 1)
        out[off : off + len(b)] = np.column_stack(
            [
                (sums / n).astype(np.float32),
                sums.astype(np.float32),
                (pm_sums / n).astype(np.float32),
                np.where(maxes < -1e8, 0, maxes).astype(np.float32),
                (hit / n).astype(np.float32),
                np.log1p(item_count[cand]).astype(np.float32),
            ]
        )
    return out


def run_cooccur(day=6):
    t0 = time.time()
    val, y, events, base = load_v20(day)
    cutoff = START + (day - 2) * DAY
    keys, counts, item_count = build_cooccur(cutoff)
    n_users = int(np.unique(train[train[:, 2] < cutoff][::10, 0]).size)
    fit = fit_window(day)
    xf = cooccur_scores(fit, hist, keys, counts, item_count, n_users)
    xv = cooccur_scores(val, hist, keys, counts, item_count, n_users)
    # within-event relatives
    def add_rel(x):
        rel = []
        for j in range(x.shape[1]):
            m = x[:, j].reshape(-1, 10)
            rel.append((m - m.mean(1, keepdims=True)).ravel())
            rel.append((np.argsort(np.argsort(m, 1), 1) / 9.0).ravel())
        return np.column_stack([x, *rel]).astype(np.float32)

    xf, xv = add_rel(xf), add_rel(xv)
    dtrain = lgb.Dataset(xf, label=fit[:, 5])
    dval = lgb.Dataset(xv, label=y, reference=dtrain)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 6,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
    )
    raw = model.predict(xv, num_iteration=model.best_iteration)
    blend = best_rank_blend(y, base, raw)
    audit_leakage("cooccur", day, cutoff, val)
    trees = int(model.best_iteration or 800)
    # day 5 confirm
    val5, y5, ev5, base5 = load_v20(5)
    cutoff5 = START + (5 - 2) * DAY
    keys5, counts5, ic5 = build_cooccur(cutoff5)
    n_users5 = int(np.unique(train[train[:, 2] < cutoff5][::10, 0]).size)
    fit5 = fit_window(5)
    xf5 = add_rel(cooccur_scores(fit5, hist, keys5, counts5, ic5, n_users5))
    xv5 = add_rel(cooccur_scores(val5, hist, keys5, counts5, ic5, n_users5))
    model5 = lgb.train(
        params,
        lgb.Dataset(xf5, label=fit5[:, 5]),
        num_boost_round=trees,
    )
    raw5 = model5.predict(xv5)
    blend5 = best_rank_blend(y5, base5, raw5, weights=[blend["weight"]])
    np.savez_compressed(ROOT / f"v21_cooccur_raw_day6.npz", y=y, events=events, raw=raw)
    np.savez_compressed(ROOT / f"v21_cooccur_raw_day5.npz", y=y5, events=ev5, raw=raw5)
    report = {
        "exp": "cooccur",
        "trees": trees,
        "day6": {
            "raw_auc": blend["raw_auc"],
            "blend_auc": blend["auc"],
            "weight": blend["weight"],
            "base_auc": blend["base_auc"],
            "delta": blend["delta"],
        },
        "day5": {
            "raw_auc": float(roc_auc_score(y5, raw5)),
            "blend_auc": blend5["auc"],
            "weight": blend["weight"],
            "base_auc": blend5["base_auc"],
            "delta": blend5["delta"],
        },
    }
    kept = blend["auc"] > blend["base_auc"] + 1e-6 and blend5["auc"] > blend5["base_auc"] + 1e-6
    if blend["delta"] > 0.01:
        print("WARNING: gain >0.01 — audit cooccur cutoff/history time safety", flush=True)
    (ROOT / "v21_cooccur_report.json").write_text(json.dumps(report, indent=2))
    if kept:
        np.savez_compressed(
            ROOT / "v21_cooccur_day6.npz", y=y, events=events, pred=blend["pred"], raw=raw
        )
        np.savez_compressed(
            ROOT / "v21_cooccur_day5.npz", y=y5, events=ev5, pred=blend5["pred"], raw=raw5
        )
    runtime = time.time() - t0
    append_log(
        "V21-2",
        "Time-safe item-item co-occurrence features (+ rank-blend vs V20)",
        blend["base_auc"],
        blend["auc"],
        runtime,
        kept,
        f"raw={blend['raw_auc']:.6f} w={blend['weight']} day5_blend={blend5['auc']:.6f}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


# ---------------------------------------------------------------------------
# Exp 3: time-gap and recency-weighted history features
# ---------------------------------------------------------------------------
def recency_features(rows, histories, times, emb):
    x = np.empty((len(rows), 14), dtype=np.float32)
    for off in range(0, len(rows), 20000):
        b = rows[off : off + 20000]
        h = np.asarray(histories[b[:, 6]])
        t = np.asarray(times[b[:, 6]])
        valid = h >= 0
        safe = np.maximum(h, 0)
        lengths = valid.sum(1)
        age_sec = np.where(valid, b[:, 2, None] - t, 0).astype(np.float64)
        age_day = age_sec / 86400.0
        # gaps
        last_t = np.where(valid, t, 0).max(1)
        gap_last = np.where(lengths > 0, (b[:, 2] - last_t) / 86400.0, 30.0)
        first_t = np.where(valid, t, np.iinfo(np.int32).max).min(1)
        span = np.where(
            lengths > 0,
            (last_t.astype(np.float64) - first_t.astype(np.float64)) / 86400.0,
            0.0,
        )
        match = h == b[:, 1, None]
        match_count = (match & valid).sum(1)
        match_age = np.where(match & valid, age_day, 999.0).min(1)
        # exp decay weights
        cols = [gap_last, span, lengths, match_count, match_age]
        for tau in (0.25, 1.0, 3.0, 7.0):
            w = np.exp(-age_day / tau) * valid
            wsum = np.maximum(w.sum(1), 1e-9)
            # similarity to candidate via emb
            sim = (emb[safe] * emb[b[:, 1], None, :]).sum(2) * valid
            cols.append(((sim * w).sum(1) / wsum))
            cols.append(((match & valid).astype(np.float64) * w).sum(1) / wsum)
        # most recent item == candidate
        newest = np.where(valid, np.arange(20), -1).max(1)
        recent_item = h[np.arange(len(b)), np.clip(newest, 0, 19)]
        recent_match = ((newest >= 0) & (recent_item == b[:, 1])).astype(np.float64)
        cols.append(recent_match)
        x[off : off + len(b)] = np.column_stack(cols).astype(np.float32)
    return x


def run_recency(day=6):
    t0 = time.time()
    val, y, events, base = load_v20(day)
    cutoff = START + (day - 2) * DAY
    emb, _ = item_embeddings(cutoff, dim=32)
    fit = fit_window(day)
    xf = recency_features(fit, hist, htimes, emb)
    xv = recency_features(val, hist, htimes, emb)

    def add_rel(x):
        rel = []
        for j in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if j >= x.shape[1]:
                continue
            m = x[:, j].reshape(-1, 10)
            rel.append((m - m.mean(1, keepdims=True)).ravel())
            rel.append((np.argsort(np.argsort(m, 1), 1) / 9.0).ravel())
        return np.column_stack([x, *rel]).astype(np.float32)

    xf, xv = add_rel(xf), add_rel(xv)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 6,
    }
    model = lgb.train(
        params,
        lgb.Dataset(xf, label=fit[:, 5]),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(xv, label=y)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
    )
    raw = model.predict(xv, num_iteration=model.best_iteration)
    blend = best_rank_blend(y, base, raw)
    audit_leakage("recency", day, cutoff, val)
    trees = int(model.best_iteration or 800)
    val5, y5, ev5, base5 = load_v20(5)
    cutoff5 = START + (5 - 2) * DAY
    emb5, _ = item_embeddings(cutoff5, dim=32)
    fit5 = fit_window(5)
    xf5 = add_rel(recency_features(fit5, hist, htimes, emb5))
    xv5 = add_rel(recency_features(val5, hist, htimes, emb5))
    model5 = lgb.train(params, lgb.Dataset(xf5, label=fit5[:, 5]), num_boost_round=trees)
    raw5 = model5.predict(xv5)
    blend5 = best_rank_blend(y5, base5, raw5, weights=[blend["weight"]])
    np.savez_compressed(ROOT / "v21_recency_raw_day6.npz", y=y, events=events, raw=raw)
    np.savez_compressed(ROOT / "v21_recency_raw_day5.npz", y=y5, events=ev5, raw=raw5)
    report = {
        "exp": "recency",
        "trees": trees,
        "day6": {
            "raw_auc": blend["raw_auc"],
            "blend_auc": blend["auc"],
            "weight": blend["weight"],
            "base_auc": blend["base_auc"],
            "delta": blend["delta"],
        },
        "day5": {
            "raw_auc": float(roc_auc_score(y5, raw5)),
            "blend_auc": blend5["auc"],
            "weight": blend["weight"],
            "base_auc": blend5["base_auc"],
            "delta": blend5["delta"],
        },
    }
    kept = blend["auc"] > blend["base_auc"] + 1e-6 and blend5["auc"] > blend5["base_auc"] + 1e-6
    if blend["delta"] > 0.01:
        print("WARNING: gain >0.01 — audit recency time gaps vs event stamp", flush=True)
    (ROOT / "v21_recency_report.json").write_text(json.dumps(report, indent=2))
    if kept:
        np.savez_compressed(
            ROOT / "v21_recency_day6.npz", y=y, events=events, pred=blend["pred"], raw=raw
        )
        np.savez_compressed(
            ROOT / "v21_recency_day5.npz", y=y5, events=ev5, pred=blend5["pred"], raw=raw5
        )
    runtime = time.time() - t0
    append_log(
        "V21-3",
        "Time-gap + recency-weighted history features (+ rank-blend vs V20)",
        blend["base_auc"],
        blend["auc"],
        runtime,
        kept,
        f"raw={blend['raw_auc']:.6f} w={blend['weight']} day5_blend={blend5['auc']:.6f}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


# ---------------------------------------------------------------------------
# Exp 4: blend any kept members (+ all raw candidates) with V20
# ---------------------------------------------------------------------------
def run_blend():
    t0 = time.time()
    val, y, events, base = load_v20(6)
    members = {"v20": base}
    for name in ("sequence", "cooccur", "recency"):
        path = ROOT / f"v21_{name}_raw_day6.npz"
        if path.exists():
            z = np.load(path)
            assert np.array_equal(z["events"], events) and np.array_equal(z["y"], y)
            members[name] = z["raw"].astype(np.float64)
            print(f"Loaded {name} raw_auc={roc_auc_score(y, members[name]):.6f}", flush=True)
    # grid search convex rank-blend weights
    ranks = {k: rankdata(v, method="average") / (len(v) + 1) for k, v in members.items()}
    keys = list(ranks.keys())
    best = {"auc": float(roc_auc_score(y, base)), "weights": {"v20": 1.0}, "pred": base}
    # coarse grid: v20 vs each other, and 3-way if multiple
    others = [k for k in keys if k != "v20"]
    for o in others:
        for w in np.linspace(0.05, 0.5, 10):
            pred = (1 - w) * ranks["v20"] + w * ranks[o]
            auc = float(roc_auc_score(y, pred))
            if auc > best["auc"]:
                best = {"auc": auc, "weights": {"v20": 1 - w, o: w}, "pred": pred}
    if len(others) >= 2:
        for w1 in np.linspace(0.0, 0.4, 5):
            for w2 in np.linspace(0.0, 0.4, 5):
                w0 = 1 - w1 - w2
                if w0 < 0.2:
                    continue
                pred = w0 * ranks["v20"] + w1 * ranks[others[0]] + w2 * ranks[others[1]]
                if len(others) > 2:
                    # leave rest at 0 for this coarse pass
                    pass
                auc = float(roc_auc_score(y, pred))
                if auc > best["auc"]:
                    best = {
                        "auc": auc,
                        "weights": {"v20": w0, others[0]: w1, others[1]: w2},
                        "pred": pred,
                    }
    if len(others) >= 3:
        for w1 in [0.05, 0.1, 0.15, 0.2]:
            for w2 in [0.05, 0.1, 0.15]:
                for w3 in [0.05, 0.1, 0.15]:
                    w0 = 1 - w1 - w2 - w3
                    if w0 < 0.4:
                        continue
                    pred = (
                        w0 * ranks["v20"]
                        + w1 * ranks[others[0]]
                        + w2 * ranks[others[1]]
                        + w3 * ranks[others[2]]
                    )
                    auc = float(roc_auc_score(y, pred))
                    if auc > best["auc"]:
                        best = {
                            "auc": auc,
                            "weights": {
                                "v20": w0,
                                others[0]: w1,
                                others[1]: w2,
                                others[2]: w3,
                            },
                            "pred": pred,
                        }

    # day 5 with frozen weights
    val5, y5, ev5, base5 = load_v20(5)
    ranks5 = {"v20": rankdata(base5, method="average") / (len(base5) + 1)}
    for name in others:
        path = ROOT / f"v21_{name}_raw_day5.npz"
        if not path.exists():
            continue
        z = np.load(path)
        assert np.array_equal(z["events"], ev5)
        ranks5[name] = rankdata(z["raw"], method="average") / (len(z["raw"]) + 1)
    pred5 = np.zeros(len(y5), dtype=np.float64)
    for k, w in best["weights"].items():
        pred5 += w * ranks5[k]
    auc5 = float(roc_auc_score(y5, pred5))
    base5_auc = float(roc_auc_score(y5, base5))
    report = {
        "exp": "blend",
        "day6": {
            "blend_auc": best["auc"],
            "base_auc": V20[6],
            "delta": best["auc"] - V20[6],
            "weights": best["weights"],
        },
        "day5": {
            "blend_auc": auc5,
            "base_auc": base5_auc,
            "delta": auc5 - base5_auc,
            "weights": best["weights"],
        },
    }
    kept = best["auc"] > V20[6] + 1e-6 and auc5 > base5_auc + 1e-6
    if report["day6"]["delta"] > 0.01:
        print("WARNING: blend gain >0.01 — inspect member leakage audits", flush=True)
    (ROOT / "v21_blend_report.json").write_text(json.dumps(report, indent=2))
    if kept:
        np.savez_compressed(
            ROOT / "v21_blend_day6.npz", y=y, events=events, pred=best["pred"]
        )
        np.savez_compressed(ROOT / "v21_blend_day5.npz", y=y5, events=ev5, pred=pred5)
        (ROOT / "v21_blend_weights.json").write_text(json.dumps(best["weights"], indent=2))
    runtime = time.time() - t0
    append_log(
        "V21-4",
        "Rank-blend new models with V20",
        V20[6],
        best["auc"],
        runtime,
        kept,
        f"w={best['weights']} day5={auc5:.6f}",
    )
    print(json.dumps(report, indent=2), flush=True)
    print("KEPT" if kept else "REJECTED", flush=True)
    return report, kept


def run_finalize():
    """Build submission_v21.csv only if blend (or a kept single model) beat V20."""
    blend_path = ROOT / "v21_blend_report.json"
    if not blend_path.exists():
        print("No blend report; nothing to finalize.", flush=True)
        return
    report = json.loads(blend_path.read_text())
    if report["day6"]["delta"] <= 0 or report["day5"]["delta"] <= 0:
        print("Blend did not beat V20 on both days; not writing submission_v21.csv", flush=True)
        append_log(
            "V21-final",
            "Finalize submission_v21",
            V20[6],
            report["day6"]["blend_auc"],
            0,
            False,
            "skipped — no both-day gain",
        )
        return
    print(
        "Finalize requires re-scoring evaluation with the same recipe; "
        "members must expose eval predictions. Building from available final raws if present.",
        flush=True,
    )
    # Prefer explicit eval predictions if experiments wrote them; otherwise refuse silently.
    weights = json.loads((ROOT / "v21_blend_weights.json").read_text())
    parts = {}
    # V20 submission as probability ranks source
    import csv as _csv

    v20_prob = np.empty(529390, dtype=np.float64)
    with (ROOT / "submission_v20.csv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(_csv.DictReader(f)):
            v20_prob[i] = float(row["predicted_probability"])
    parts["v20"] = v20_prob
    for name in ("sequence", "cooccur", "recency"):
        p = ROOT / f"v21_{name}_eval.npy"
        if p.exists() and name in weights:
            parts[name] = np.load(p)
    missing = [k for k in weights if k not in parts]
    if missing:
        print(f"Missing eval predictions for {missing}; run member final fits first.", flush=True)
        append_log(
            "V21-final",
            "Finalize submission_v21",
            V20[6],
            report["day6"]["blend_auc"],
            0,
            False,
            f"missing eval preds {missing}",
        )
        return
    ranks = {k: rankdata(v, method="average") / (len(v) + 1) for k, v in parts.items()}
    pred = np.zeros(529390, dtype=np.float64)
    for k, w in weights.items():
        pred += w * ranks[k]
    pred = np.clip(pred, 0, 1)
    out = Path("submission_v21.csv")
    with EVALUATION_CSV.open(newline="", encoding="utf-8") as src, out.open(
        "w", newline="", encoding="utf-8"
    ) as dest:
        w = csv.writer(dest)
        w.writerow(["rowid", "predicted_probability"])
        for row, p in zip(csv.DictReader(src), pred, strict=True):
            w.writerow([row["rowid"], format(float(p), ".10g")])
    (ROOT / "submission_v21.csv").write_bytes(out.read_bytes())
    print(f"Wrote {out} — submission.csv left unchanged", flush=True)
    append_log(
        "V21-final",
        "Wrote submission_v21.csv (submission.csv untouched)",
        V20[6],
        report["day6"]["blend_auc"],
        0,
        True,
        f"weights={weights}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        required=True,
        choices=["sequence", "cooccur", "recency", "blend", "finalize"],
    )
    args = ap.parse_args()
    if args.exp == "sequence":
        run_sequence()
    elif args.exp == "cooccur":
        run_cooccur()
    elif args.exp == "recency":
        run_recency()
    elif args.exp == "blend":
        run_blend()
    else:
        run_finalize()


if __name__ == "__main__":
    main()
