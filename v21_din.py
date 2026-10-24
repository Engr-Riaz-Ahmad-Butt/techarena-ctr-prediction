"""
V21 DIN-style neural CTR model.

Steps:
  python v21_din.py --step debug          # sanity + tiny overfit
  python v21_din.py --step train --init als
  python v21_din.py --step train --init random
  python v21_din.py --step compare        # pick better init on day6/day5
  python v21_din.py --step oof            # time-safe OOF scores for stacking
  python v21_din.py --step eval_pred      # evaluation predictions (best init)

Does not touch submission.csv or V20 artifacts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as ROOT

SEED = 42
DAY = 86400
V20 = {6: 0.8215282579633503, 5: 0.814599936313616}
LOG = Path("results_log.md")

torch.manual_seed(SEED)
np.random.seed(SEED)

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
PAD = NI  # padding index for histories


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
    return val, ref["y"].astype(np.float64), ref["events"], ref["pred"].astype(np.float64)


def fit_rows(day: int):
    """All labeled impressions in [day-3, day) — more data for the neural net."""
    return np.asarray(
        train[(train[:, 2] >= START + (day - 3) * DAY) & (train[:, 2] < START + day * DAY)]
    )


# ---------------------------------------------------------------------------
# Event tensor packing (shared history per event of 10 candidates)
# ---------------------------------------------------------------------------
def pack_events(rows, histories, times):
    """Return arrays shaped (n_events, ...). Vectorized left-padding."""
    assert len(rows) % 10 == 0
    n = len(rows) // 10
    block = rows.reshape(n, 10, 7)
    head = block[:, 0, :]
    h = np.asarray(histories[head[:, 6]])  # (n, 20), oldest→newest, -1 pad on the right
    t = np.asarray(times[head[:, 6]]).astype(np.int64)
    valid = h >= 0
    if np.any(valid & (t >= head[:, 2, None])):
        raise ValueError("Click history must be strictly earlier than its event")
    lengths = valid.sum(1).astype(np.int32)
    # Left-pad: move valid prefix to the right end so index 19 is most recent
    packed = np.full((n, 20), PAD, dtype=np.int64)
    gaps = np.zeros((n, 20), dtype=np.float32)
    event_ts = head[:, 2].astype(np.int64)
    # position in packed: dest_j = 20 - L + src_j for src_j in 0..L-1
    src = np.arange(20)[None, :]
    dest = src + (20 - lengths)[:, None]
    take = src < lengths[:, None]
    # fancy index rows
    rows_i = np.repeat(np.arange(n), lengths)
    src_j = np.concatenate([np.arange(L) for L in lengths]) if n else np.array([], dtype=np.int64)
    if len(src_j):
        dest_j = src_j + (20 - lengths[rows_i])
        packed[rows_i, dest_j] = h[rows_i, src_j]
        raw_gaps = (event_ts[rows_i] - t[rows_i, src_j]) / 86400.0
        gaps[rows_i, dest_j] = np.clip(raw_gaps, 0, 60).astype(np.float32)
    mask = packed != PAD
    # sanity on a sample
    if n:
        assert mask.sum(1).max() <= 20
    cands = block[:, :, 1].astype(np.int64)
    labels = block[:, :, 5].astype(np.float32)
    bucket = head[:, 3].astype(np.int64)
    day_type = head[:, 4].astype(np.int64)
    has = labels.sum(1) > 0
    target = np.where(has, labels.argmax(1), -1).astype(np.int64)
    return {
        "hist": packed,
        "mask": mask,
        "gaps": gaps,
        "cands": cands,
        "labels": labels,
        "target": target,
        "has_click": has,
        "bucket": bucket,
        "day_type": day_type,
        "n": n,
    }


class DIN(nn.Module):
    """Candidate-aware attention over click history (DIN-style) + event softmax."""

    def __init__(self, n_items: int, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.pad = n_items
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.gap_emb = nn.Linear(1, dim)
        self.attn = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.PReLU(),
            nn.Linear(dim, 1),
        )
        self.bucket = nn.Embedding(4, 8)
        self.daytype = nn.Embedding(2, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 8 + 4 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, hist, mask, gaps, cands, bucket, day_type):
        # hist: B,20 ; cands: B,10 ; mask: B,20 bool
        he = self.item(hist) + self.gap_emb(gaps.unsqueeze(-1))  # B,20,D
        ce = self.item(cands.clamp(max=self.pad))  # B,10,D  (cands never pad)
        b = self.bucket(bucket)
        d = self.daytype(day_type)
        logits = []
        for j in range(10):
            cj = ce[:, j : j + 1, :].expand(-1, 20, -1)
            cat = torch.cat([he, cj, he * cj, he - cj], dim=-1)
            score = self.attn(cat).squeeze(-1)
            score = score.masked_fill(~mask, -1e9)
            # if all masked, softmax is uniform nan — guard
            all_pad = ~mask.any(dim=1)
            score = score.masked_fill(all_pad.unsqueeze(1), 0.0)
            a = torch.softmax(score, dim=-1)
            a = a.masked_fill(all_pad.unsqueeze(1), 0.0)
            ctx = (a.unsqueeze(-1) * he).sum(1)
            empty = all_pad.float().unsqueeze(1)
            feat = torch.cat([ctx, ce[:, j], b, d, empty], dim=-1)
            logits.append(self.mlp(feat).squeeze(-1))
        return torch.stack(logits, dim=1)  # B,10


def load_als_item_factors(cutoff_day: int, dim: int):
    """Load V item factors from the latest ALS checkpoint strictly before fit window.

    For validation day D, fit uses data up to D; ALS cut (D-2) or nearest available
    with cutoff timestamp < START + D*DAY.
    """
    # Prefer cut = day-2 style checkpoints used in the stack
    candidates = [
        ROOT / f"imp17_als_cut{cutoff_day}.npz",
        ROOT / f"imp15_model_day{cutoff_day}.npz",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        # fall back to cut4 / cut3
        for c in range(cutoff_day - 1, 0, -1):
            p = ROOT / f"imp17_als_cut{c}.npz"
            if p.exists():
                path = p
                break
    assert path is not None, "No ALS checkpoint found"
    z = np.load(path)
    if int(z["cutoff"]) > START + cutoff_day * DAY:
        raise ValueError("ALS checkpoint contains data after the requested cutoff")
    V = z["V"].astype(np.float32)
    assert V.shape[0] == NI
    if V.shape[1] >= dim:
        V = V[:, :dim]
    else:
        pad = np.zeros((NI, dim - V.shape[1]), dtype=np.float32)
        V = np.concatenate([V, pad], axis=1)
    # L2 normalize rows for stable init
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(nrm, 1e-8)
    print(f"ALS init from {path.name} shape={V.shape} cutoff_meta={int(z['cutoff'])}", flush=True)
    return V


def init_embeddings(model: DIN, mode: str, als_cut: int, dim: int):
    with torch.no_grad():
        if mode == "als":
            V = load_als_item_factors(als_cut, dim)
            model.item.weight[:-1].copy_(torch.from_numpy(V))
            model.item.weight[-1].zero_()
        elif mode == "random":
            nn.init.normal_(model.item.weight, std=0.01)
            model.item.weight[-1].zero_()
        else:
            raise ValueError(mode)


def event_batches(data, idx, bs=256, shuffle=True, rng=None):
    rng = rng or np.random.default_rng(SEED)
    eids = idx.copy()
    if shuffle:
        rng.shuffle(eids)
    for i in range(0, len(eids), bs):
        sl = eids[i : i + bs]
        yield {k: (torch.from_numpy(v[sl]) if isinstance(v, np.ndarray) else v) for k, v in [
            ("hist", data["hist"]),
            ("mask", data["mask"]),
            ("gaps", data["gaps"]),
            ("cands", data["cands"]),
            ("bucket", data["bucket"]),
            ("day_type", data["day_type"]),
            ("target", data["target"]),
            ("has_click", data["has_click"]),
            ("labels", data["labels"]),
        ]}


def batch_loss(model, batch):
    logits = model(
        batch["hist"],
        batch["mask"],
        batch["gaps"],
        batch["cands"],
        batch["bucket"],
        batch["day_type"],
    )
    has = batch["has_click"]
    loss = logits.new_zeros(())
    if has.any():
        loss = loss + nn.functional.cross_entropy(logits[has], batch["target"][has])
    if (~has).any():
        # no-click events: push probability mass down
        loss = loss + 0.05 * torch.softmax(logits[~has], dim=1).pow(2).mean()
    return loss, logits


@torch.no_grad()
def predict_logits(model, data, bs=512):
    model.eval()
    out = np.empty((data["n"], 10), dtype=np.float32)
    for i in range(0, data["n"], bs):
        sl = slice(i, min(i + bs, data["n"]))
        logits = model(
            torch.from_numpy(data["hist"][sl]),
            torch.from_numpy(data["mask"][sl]),
            torch.from_numpy(data["gaps"][sl]),
            torch.from_numpy(data["cands"][sl]),
            torch.from_numpy(data["bucket"][sl]),
            torch.from_numpy(data["day_type"][sl]),
        )
        out[sl] = logits.numpy()
    return out.ravel()


@torch.no_grad()
def predict_probs(model, data, bs=512):
    logits = predict_logits(model, data, bs=bs).reshape(-1, 10)
    # numerically stable softmax
    x = logits - logits.max(1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(1, keepdims=True)
    return p.ravel()


def debug():
    """Check alignments and overfit a tiny subset."""
    t0 = time.time()
    print("=== DIN debug ===", flush=True)
    val, y, events, base = load_v20(6)
    assert len(val) % 10 == 0
    # item ids in range
    assert int(train[:, 1].min()) >= 0 and int(train[:, 1].max()) < NI
    assert int(hist[hist >= 0].max()) < NI
    # history before event
    h = np.asarray(hist[val[::10, 6]])
    t = np.asarray(htimes[val[::10, 6]])
    valid = h >= 0
    assert np.all((t < val[::10, 2, None]) | ~valid)
    # pack
    data = pack_events(val[:1000], hist, htimes)  # 100 events
    assert data["cands"].shape == (100, 10)
    assert data["hist"].shape == (100, 20)
    # most-recent slot equals last valid history item
    for i in range(100):
        L = int(data["mask"][i].sum())
        if L == 0:
            continue
        raw_h = h[i]
        raw_h = raw_h[raw_h >= 0]
        assert data["hist"][i, -1] == raw_h[-1]
        assert np.all(data["hist"][i, -L:] == raw_h)
    # labels line up with row clicked flags
    assert np.allclose(data["labels"].ravel(), val[:1000, 5])
    print("Alignment checks PASSED", flush=True)

    # tiny overfit: 32 events with a click
    fit = fit_rows(6)
    packed = pack_events(fit, hist, htimes)
    click_idx = np.where(packed["has_click"])[0][:64]
    tiny = {k: (v[click_idx] if isinstance(v, np.ndarray) and k != "n" else v) for k, v in packed.items()}
    tiny["n"] = len(click_idx)
    model = DIN(NI, dim=32)
    init_embeddings(model, "random", als_cut=4, dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for epoch in range(80):
        model.train()
        batch = {
            "hist": torch.from_numpy(tiny["hist"]),
            "mask": torch.from_numpy(tiny["mask"]),
            "gaps": torch.from_numpy(tiny["gaps"]),
            "cands": torch.from_numpy(tiny["cands"]),
            "bucket": torch.from_numpy(tiny["bucket"]),
            "day_type": torch.from_numpy(tiny["day_type"]),
            "target": torch.from_numpy(tiny["target"]),
            "has_click": torch.from_numpy(tiny["has_click"]),
            "labels": torch.from_numpy(tiny["labels"]),
        }
        opt.zero_grad()
        loss, logits = batch_loss(model, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if epoch % 20 == 0 or epoch == 79:
            with torch.no_grad():
                pred = torch.softmax(logits, 1).argmax(1).numpy()
                acc = float((pred == tiny["target"]).mean())
            print(f"overfit epoch {epoch} loss={losses[-1]:.4f} acc={acc:.3f}", flush=True)
    assert losses[-1] < losses[0] * 0.5, "Loss did not drop enough on tiny subset"
    assert losses[-1] < 0.5, f"Overfit failed, final loss={losses[-1]}"
    print("Tiny overfit PASSED", flush=True)
    report = {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "runtime_s": time.time() - t0,
    }
    (ROOT / "v21_din_debug.json").write_text(json.dumps(report, indent=2))
    append_log(
        "V21-DIN-debug",
        "DIN debug + tiny overfit sanity",
        0.0,
        0.0,
        report["runtime_s"],
        True,
        f"loss {losses[0]:.3f}->{losses[-1]:.3f}",
    )
    return report


def train_one(
    day: int,
    init: str,
    dim: int = 64,
    max_epochs: int = 12,
    bs: int = 256,
    click_only: bool = True,
    lr: float = 2e-3,
):
    t0 = time.time()
    val, y, events, base = load_v20(day)
    fit = fit_rows(day)
    print(
        f"Packing fit={len(fit)} val={len(val)} init={init} click_only={click_only} lr={lr}",
        flush=True,
    )
    train_data = pack_events(fit, hist, htimes)
    val_data = pack_events(val, hist, htimes)
    model = DIN(NI, dim=dim)
    als_cut = max(day - 2, 2)
    init_embeddings(model, init, als_cut=als_cut, dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(SEED + day)
    best_auc, best_state, best_epoch, best_raw = -1.0, None, 0, None
    history = []
    patience, bad = 4, 0
    if click_only:
        idx = np.where(train_data["has_click"])[0]
        print(f"Using {len(idx)} click events of {train_data['n']}", flush=True)
    else:
        idx = np.arange(train_data["n"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in event_batches(train_data, idx, bs=bs, shuffle=True, rng=rng):
            batch["mask"] = batch["mask"].bool()
            batch["has_click"] = batch["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(model, batch)
            if float(loss.detach()) == 0.0:
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * batch["hist"].size(0)
            n += batch["hist"].size(0)
        raw = predict_probs(model, val_data, bs=512)
        auc = float(roc_auc_score(y, raw))
        tgt = val_data["target"]
        has = val_data["has_click"]
        acc = float((raw.reshape(-1, 10).argmax(1) == tgt)[has].mean()) if has.any() else 0.0
        row = {"epoch": epoch, "loss": total / max(n, 1), "auc": auc, "click_acc": acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if auc > best_auc + 1e-5:
            best_auc, best_epoch, best_raw = auc, epoch, raw
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break
    assert best_state is not None and best_raw is not None
    model.load_state_dict(best_state)
    out = ROOT / f"v21_din_{init}_day{day}"
    torch.save(best_state, str(out) + ".pt")
    np.savez_compressed(str(out) + "_raw.npz", y=y, events=events, raw=best_raw, base=base)
    report = {
        "day": day,
        "init": init,
        "dim": dim,
        "click_only": click_only,
        "lr": lr,
        "best_epoch": best_epoch,
        "raw_auc": best_auc,
        "v20_auc": float(roc_auc_score(y, base)),
        "history": history,
        "runtime_s": time.time() - t0,
    }
    (ROOT / f"v21_din_{init}_day{day}_report.json").write_text(json.dumps(report, indent=2))
    append_log(
        f"V21-DIN-{init}-d{day}",
        f"DIN train init={init} day{day} click_only={click_only}",
        report["v20_auc"],
        best_auc,
        report["runtime_s"],
        best_auc > report["v20_auc"],
        f"epoch={best_epoch} dim={dim} lr={lr}",
    )
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2), flush=True)
    return report


def compare_inits():
    """Require both inits trained on day 6; confirm winner on day 5."""
    reps = {}
    for init in ("als", "random"):
        p = ROOT / f"v21_din_{init}_day6_report.json"
        assert p.exists(), f"Missing {p}; run train first"
        reps[init] = json.loads(p.read_text())
    winner = max(reps.keys(), key=lambda k: reps[k]["raw_auc"])
    print(f"Day6 winner={winner} auc={reps[winner]['raw_auc']:.6f}", flush=True)
    # ensure day5 exists for winner
    p5 = ROOT / f"v21_din_{winner}_day5_report.json"
    if not p5.exists():
        train_one(5, winner, dim=reps[winner]["dim"])
        p5 = ROOT / f"v21_din_{winner}_day5_report.json"
    r5 = json.loads(p5.read_text())
    out = {
        "day6": {k: reps[k]["raw_auc"] for k in reps},
        "winner": winner,
        "day5_winner_auc": r5["raw_auc"],
        "day5_v20": r5["v20_auc"],
    }
    (ROOT / "v21_din_compare.json").write_text(json.dumps(out, indent=2))
    append_log(
        "V21-DIN-compare",
        f"DIN init compare winner={winner}",
        V20[6],
        reps[winner]["raw_auc"],
        0,
        reps[winner]["raw_auc"] > 0.7,
        json.dumps(out),
    )
    print(json.dumps(out, indent=2), flush=True)
    return out


def oof_scores():
    """
    Time-safe OOF neural scores on the day-6 / day-5 validation cohorts
    (already produced by train_one). Also build rolling OOF on fit windows
    by training on earlier days only — for stacking into trees.

    For stacking features on validation day D we use the model trained with
    cutoff before D (already time-safe). For training rows of the tree model
    we generate scores from a model whose fit window ends before those rows.
    """
    # Validation OOF = existing raw predictions
    for day in (6, 5):
        # prefer als if compare chose it
        compare = ROOT / "v21_din_compare.json"
        init = "als"
        if compare.exists():
            init = json.loads(compare.read_text())["winner"]
        src = ROOT / f"v21_din_{init}_day{day}_raw.npz"
        assert src.exists(), src
        z = np.load(src)
        np.savez_compressed(
            ROOT / f"v21_din_oof_val_day{day}.npz",
            y=z["y"],
            events=z["events"],
            raw=z["raw"],
        )
        print(f"OOF val day{day} auc={roc_auc_score(z['y'], z['raw']):.6f}", flush=True)
    # Fit-window OOF for tree training on day6 setup: score fit_rows(6) with model
    # trained for day5 (fit uses days 2-4, before day5-6) — approximate causal score.
    init = json.loads((ROOT / "v21_din_compare.json").read_text())["winner"]
    dim = json.loads((ROOT / f"v21_din_{init}_day6_report.json").read_text())["dim"]
    state = torch.load(ROOT / f"v21_din_{init}_day5.pt", map_location="cpu", weights_only=True)
    model = DIN(NI, dim=dim)
    model.load_state_dict(state)
    model.eval()
    fit = fit_rows(6)  # days 3,4,5 relative... fit_rows(6)=[3,6) i.e. days 3,4,5
    # Model day5 trained on fit_rows(5)=[2,5). Overlap on days 3,4 — not pure OOF.
    # Stricter: train a cut4 model for scoring day5+ rows only.
    print("Training cut-day4 DIN for strict OOF on later fit rows...", flush=True)
    # Train on [START+1*DAY, START+4*DAY) validate briefly unused
    rows = np.asarray(train[(train[:, 2] >= START + 1 * DAY) & (train[:, 2] < START + 4 * DAY)])
    data = pack_events(rows, hist, htimes)
    m4 = DIN(NI, dim=dim)
    init_embeddings(m4, init, als_cut=2, dim=dim)
    opt = torch.optim.Adam(m4.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    idx = np.arange(data["n"])
    for epoch in range(1, 8):
        m4.train()
        total = 0.0
        n = 0
        for batch in event_batches(data, idx, bs=256, shuffle=True, rng=rng):
            batch["mask"] = batch["mask"].bool()
            batch["has_click"] = batch["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(m4, batch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch["hist"].size(0)
            n += batch["hist"].size(0)
        print(f"oof-cut4 epoch {epoch} loss={total/max(n,1):.4f}", flush=True)
    torch.save(m4.state_dict(), ROOT / f"v21_din_{init}_cut4_oof.pt")
    # Score all of fit_rows(6); for rows with timestamp < START+4*DAY this is still
    # partially in-train — tree stacking should only use scores where ts >= START+4*DAY
    # OR we accept mild leakage within fit and rely on val being clean.
    # Clean approach: score only events with ts >= START+4*DAY using cut4 model.
    later = np.asarray(train[(train[:, 2] >= START + 4 * DAY) & (train[:, 2] < START + 6 * DAY)])
    later_data = pack_events(later, hist, htimes)
    raw_later = predict_probs(m4, later_data)
    np.savez_compressed(
        ROOT / "v21_din_oof_fit_day456.npz",
        rows_timestamp_start=int(START + 4 * DAY),
        rows_timestamp_end=int(START + 6 * DAY),
        raw=raw_later,
        y=later[:, 5],
        events=later[:, 6],
    )
    print(
        f"Strict OOF fit rows={len(later)} auc={roc_auc_score(later[:,5], raw_later):.6f}",
        flush=True,
    )
    append_log(
        "V21-DIN-oof",
        "DIN time-safe OOF scores for stacking",
        0.0,
        float(roc_auc_score(later[:, 5], raw_later)),
        0,
        True,
        f"init={init} fit_oof_rows={len(later)}",
    )


def eval_pred():
    """Train final DIN on days 4-6 window (mirroring finalize) and score evaluation."""
    compare = json.loads((ROOT / "v21_din_compare.json").read_text())
    init = compare["winner"]
    dim = json.loads((ROOT / f"v21_din_{init}_day6_report.json").read_text())["dim"]
    # Final fit: all events from day 3 onward (same spirit as stack final)
    rows = np.asarray(train[train[:, 2] >= START + 3 * DAY])
    data = pack_events(rows, hist, htimes)
    model = DIN(NI, dim=dim)
    init_embeddings(model, init, als_cut=4, dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    idx = np.arange(data["n"])
    epochs = json.loads((ROOT / f"v21_din_{init}_day6_report.json").read_text())["best_epoch"]
    epochs = max(epochs, 3)
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in event_batches(data, idx, bs=256, shuffle=True, rng=rng):
            batch["mask"] = batch["mask"].bool()
            batch["has_click"] = batch["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(model, batch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch["hist"].size(0)
            n += batch["hist"].size(0)
        print(f"final epoch {epoch}/{epochs} loss={total/max(n,1):.4f}", flush=True)
    torch.save(model.state_dict(), ROOT / f"v21_din_{init}_final.pt")
    edata = pack_events(np.asarray(evaluation), ehist, etimes)
    raw = predict_probs(model, edata)
    assert len(raw) == 529390 and np.isfinite(raw).all()
    np.save(ROOT / "v21_din_eval.npy", raw.astype(np.float32))
    print("Wrote v21_din_eval.npy", flush=True)
    append_log(
        "V21-DIN-eval",
        "DIN evaluation predictions",
        0.0,
        float(raw.mean()),
        0,
        True,
        f"init={init} epochs={epochs} mean_p={float(raw.mean()):.6f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--step",
        required=True,
        choices=["debug", "train", "compare", "oof", "eval_pred"],
    )
    ap.add_argument("--init", choices=["als", "random"], default="als")
    ap.add_argument("--day", type=int, default=6)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--all-events", action="store_true", help="Include no-click events")
    args = ap.parse_args()
    if args.step == "debug":
        debug()
    elif args.step == "train":
        train_one(
            args.day,
            args.init,
            dim=args.dim,
            max_epochs=args.epochs,
            click_only=not args.all_events,
            lr=args.lr,
        )
    elif args.step == "compare":
        compare_inits()
    elif args.step == "oof":
        oof_scores()
    else:
        eval_pred()


if __name__ == "__main__":
    main()
