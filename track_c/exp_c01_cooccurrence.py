"""Track C Experiment 01: Full-prefix item-item co-occurrence features.

Hypothesis: Building co-occurrence statistics from ALL clicks a user made before
the cutoff (not just the 20-item history window) may capture broader item similarity
patterns that V53's ALS factors miss. Especially helpful for sparse-history users
where ALS factors are poorly estimated.

Features:
  1. For each candidate, mean/max PPMI with history items
  2. For each candidate, mean/max cosine similarity with history items
  3. Candidate item global frequency (log-transformed)
  4. Candidate-in-history flag (was this item clicked before?)
  5. Slate-relative: candidate co-occurrence score minus slate mean
  6. V53 baseline logit (to learn residual corrections)

Model: sklearn HistGradientBoostingClassifier (fast, CPU-friendly, no GPU needed)
Blend: rank-average with V53 at several weights, pick best on selection day 6.

Temporal safety: co-occurrence graph uses only clicks strictly before the cutoff.
"""
import gc
import json
import time
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def rank(x):
    return rankdata(np.asarray(x, dtype="float64"), method="average") / (len(x) + 1)


def build_prefix_ledger(train, hist, hist_times, cutoff_ts):
    """Build a ledger of all (timestamp, item) clicks per user before cutoff."""
    ledger = defaultdict(set)
    n_events = len(train) // 10
    for ev_idx in range(n_events):
        row0 = ev_idx * 10
        ts = int(train[row0, 2])
        if ts >= cutoff_ts:
            continue
        uid = int(train[row0, 0])
        # Add clicks from history (prior to this event)
        for item, stamp in zip(hist[ev_idx], hist_times[ev_idx]):
            if item >= 0:
                ledger[uid].add((int(stamp), int(item)))
        # Add the click from this event (if any)
        for slot in range(10):
            row = row0 + slot
            if int(train[row, 5]) == 1:
                ledger[uid].add((ts, int(train[row, 1])))
    return dict(ledger)


def build_graph(ledger, n_items):
    """Build sparse item-item co-occurrence matrix from binary user-item profiles."""
    profiles = []
    for uid, clicks in ledger.items():
        items = sorted(set(item for _, item in clicks))
        if len(items) > 1:
            profiles.append(items)

    if not profiles:
        return sparse.csr_matrix((n_items, n_items)), np.zeros(n_items), 0

    lengths = [len(p) for p in profiles]
    rows = np.repeat(np.arange(len(profiles)), lengths)
    cols = np.concatenate(profiles)
    x = sparse.csr_matrix(
        (np.ones(len(rows), dtype="float32"), (rows, cols)),
        shape=(len(profiles), n_items),
    )
    frequency = np.asarray(x.sum(0)).ravel()
    support = (x.T @ x).tocsr()
    support.setdiag(0)
    support.eliminate_zeros()
    del x, rows, cols
    gc.collect()
    return support, frequency, len(profiles)


def extract_features(support, frequency, n_users, train, hist, hist_times, event_indices, cutoff_ts, ledger):
    """Compute co-occurrence features for validation events."""
    n_items = len(frequency)
    n_events = len(event_indices)
    n_rows = n_events * 10

    # Get candidate items and history for each event
    out = np.zeros((n_rows, 14), dtype="float32")

    for batch_start in range(0, n_events, 256):
        batch_end = min(batch_start + 256, n_events)
        batch_ev = event_indices[batch_start:batch_end]
        batch_size = len(batch_ev)

        # Get candidate items (10 per event)
        candidates = np.zeros((batch_size, 10), dtype="int64")
        for i, ev in enumerate(batch_ev):
            for slot in range(10):
                candidates[i, slot] = int(train[ev * 10 + slot, 1])

        # Get history items from FULL PREFIX (up to 100 most recent)
        history = np.full((batch_size, 100), -1, dtype="int32")
        for i, ev in enumerate(batch_ev):
            uid = int(train[ev * 10, 0])
            user_clicks = ledger.get(uid, set())
            # Only clicks before this event's timestamp
            ev_ts = int(train[ev * 10, 2])
            prior = sorted(
                [(ts, item) for ts, item in user_clicks if ts < ev_ts],
                reverse=True,
            )[:100]
            for j, (_, item) in enumerate(prior):
                history[i, j] = item

        # Compute features for this batch
        for i in range(batch_size):
            h_items = history[i][history[i] >= 0]
            h_len = len(h_items)

            for slot in range(10):
                row_out = (batch_start + i) * 10 + slot
                cand = candidates[i, slot]

                if h_len == 0:
                    # No history - just item frequency
                    out[row_out, 0:6] = 0.0
                    out[row_out, 6] = np.log1p(frequency[cand]) if cand < n_items else 0.0
                    out[row_out, 7] = 0  # not in history
                    out[row_out, 8] = h_len
                    continue

                # Co-occurrence values between candidate and each history item
                cooc_vals = np.array(
                    [float(support[cand, h]) if cand < n_items and h < n_items else 0.0 for h in h_items]
                )

                # PPMI
                cand_freq = max(frequency[cand], 1) if cand < n_items else 1
                h_freqs = np.array([max(frequency[h], 1) if h < n_items else 1 for h in h_items])
                pmi = np.log(np.maximum(cooc_vals * n_users / (cand_freq * h_freqs), 1e-10))
                ppmi = np.maximum(pmi, 0) * cooc_vals / (cooc_vals + 5)

                # Cosine similarity
                cosine = cooc_vals / np.sqrt(np.maximum(cand_freq * h_freqs, 1))

                # Log co-occurrence
                log_cooc = np.log1p(cooc_vals)

                # Features
                out[row_out, 0] = ppmi.mean()
                out[row_out, 1] = ppmi.max() if len(ppmi) > 0 else 0
                out[row_out, 2] = cosine.mean()
                out[row_out, 3] = cosine.max() if len(cosine) > 0 else 0
                out[row_out, 4] = log_cooc.mean()
                out[row_out, 5] = log_cooc.max() if len(log_cooc) > 0 else 0
                out[row_out, 6] = np.log1p(frequency[cand]) if cand < n_items else 0.0
                out[row_out, 7] = 1.0 if cand in set(h_items) else 0.0
                out[row_out, 8] = h_len
                # Co-occurrence support count (how many history items have nonzero cooc)
                out[row_out, 9] = (cooc_vals > 0).sum() / max(h_len, 1)

    # Slate-relative features (computed after all events)
    for ev in range(n_events):
        block = out[ev * 10 : (ev + 1) * 10]
        for col in [0, 2, 4]:  # ppmi_mean, cosine_mean, log_cooc_mean
            slate_mean = block[:, col].mean()
            out[ev * 10 : (ev + 1) * 10, 10 + col // 2] = block[:, col] - slate_mean
        # Rank within slate by ppmi_mean
        if block[:, 0].std() > 1e-9:
            out[ev * 10 : (ev + 1) * 10, 13] = rankdata(block[:, 0]) / 11.0

    return out


def main():
    start = time.monotonic()
    print("=" * 60)
    print("Track C Experiment 01: Full-prefix co-occurrence features")
    print("=" * 60)

    # Load data
    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    hist = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    hist_times = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")

    START_TS = int(train[0, 2])
    DAY = 86400
    n_items = int(train[:, 1].max()) + 1
    n_events_total = len(train) // 10

    # Load V53 baseline predictions
    z5 = np.load(ARTIFACTS / "v53_causal_propensity_day5.npz")
    z6 = np.load(ARTIFACTS / "v53_causal_propensity_day6.npz")

    # Get first-event indices for each validation day
    def first_event_indices(day):
        cutoff = START_TS + day * DAY
        later = np.asarray(train[(train[:, 2] >= cutoff) & (train[:, 2] < cutoff + DAY)])
        events = later[::10]
        _, first = np.unique(events[:, 0], return_index=True)
        return events[first, 6]  # event indices in the original train array

    # --- DAY 5 ---
    print(f"\n[{time.monotonic()-start:.0f}s] Building prefix ledger for Day 5...")
    cutoff5 = START_TS + 5 * DAY
    ledger5 = build_prefix_ledger(train, hist, hist_times, cutoff5)
    print(f"  Ledger: {len(ledger5)} users")

    print(f"[{time.monotonic()-start:.0f}s] Building co-occurrence graph for Day 5...")
    support5, freq5, n_users5 = build_graph(ledger5, n_items)
    print(f"  Graph: {n_users5} users, {support5.nnz} non-zero pairs")

    ev5 = z5["events"][::10]  # event indices
    print(f"[{time.monotonic()-start:.0f}s] Extracting features for Day 5 ({len(ev5)} events)...")
    feat5 = extract_features(support5, freq5, n_users5, train, hist, hist_times, ev5, cutoff5, ledger5)
    print(f"  Feature shape: {feat5.shape}")

    # Add V53 baseline logit as a feature
    v53_logit5 = logit(np.clip(z5["pred"], 1e-6, 1 - 1e-6))
    X5 = np.column_stack([feat5, v53_logit5])
    y5 = z5["y"].copy()

    del support5, ledger5
    gc.collect()

    # --- DAY 6 ---
    print(f"\n[{time.monotonic()-start:.0f}s] Building prefix ledger for Day 6...")
    cutoff6 = START_TS + 6 * DAY
    ledger6 = build_prefix_ledger(train, hist, hist_times, cutoff6)
    print(f"  Ledger: {len(ledger6)} users")

    print(f"[{time.monotonic()-start:.0f}s] Building co-occurrence graph for Day 6...")
    support6, freq6, n_users6 = build_graph(ledger6, n_items)
    print(f"  Graph: {n_users6} users, {support6.nnz} non-zero pairs")

    ev6 = z6["events"][::10]
    print(f"[{time.monotonic()-start:.0f}s] Extracting features for Day 6 ({len(ev6)} events)...")
    feat6 = extract_features(support6, freq6, n_users6, train, hist, hist_times, ev6, cutoff6, ledger6)
    print(f"  Feature shape: {feat6.shape}")

    v53_logit6 = logit(np.clip(z6["pred"], 1e-6, 1 - 1e-6))
    X6 = np.column_stack([feat6, v53_logit6])
    y6 = z6["y"].copy()

    del support6, ledger6
    gc.collect()

    # --- TRAIN on Day 5, predict Day 6 ---
    print(f"\n[{time.monotonic()-start:.0f}s] Training LightGBM on Day 5...")
    params = dict(
        objective="binary",
        metric="auc",
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=300,
        min_child_samples=200,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=2,
        random_state=7001,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X5, y5)
    raw6 = model.predict_proba(X6)[:, 1]

    standalone_auc6 = roc_auc_score(y6, raw6)
    print(f"  Standalone Day6 ordinary AUC: {standalone_auc6:.6f}")

    # --- Also train on Day 5 with user cross-fitting for Day 5 OOF predictions ---
    print(f"\n[{time.monotonic()-start:.0f}s] Cross-fitting Day 5 (2 user folds)...")
    users5 = train[z5["events"], 0]
    unique_users5 = np.unique(users5)
    np.random.seed(7002)
    np.random.shuffle(unique_users5)
    mid = len(unique_users5) // 2
    fold_a_users = set(unique_users5[:mid])
    fold_a_mask = np.array([int(u) in fold_a_users for u in users5])
    fold_b_mask = ~fold_a_mask

    raw5_oof = np.zeros(len(y5), dtype="float64")
    for fold_name, fit_mask, pred_mask in [("A", fold_a_mask, fold_b_mask), ("B", fold_b_mask, fold_a_mask)]:
        m = lgb.LGBMClassifier(**params)
        m.fit(X5[fit_mask], y5[fit_mask])
        raw5_oof[pred_mask] = m.predict_proba(X5[pred_mask])[:, 1]
        print(f"  Fold {fold_name}: fit {fit_mask.sum()} rows, predict {pred_mask.sum()} rows")

    standalone_auc5 = roc_auc_score(y5, raw5_oof)
    print(f"  Standalone Day5 OOF ordinary AUC: {standalone_auc5:.6f}")

    # --- Evaluate blends ---
    print(f"\n[{time.monotonic()-start:.0f}s] Evaluating rank blends with V53...")

    # Load evaluation weights for adjusted AUC
    eh = np.load(ARTIFACTS / "evaluation_hist.npy", mmap_mode="r")
    h = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)

    def weighted_auc(events, y, pred):
        vg = np.digitize((h[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
        w = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
        return float(roc_auc_score(y, pred, sample_weight=w))

    baseline_adj5 = weighted_auc(z5["events"], y5, z5["pred"])
    baseline_adj6 = weighted_auc(z6["events"], y6, z6["pred"])

    results = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred5_blend = (1 - w) * rank(z5["pred"]) + w * rank(raw5_oof)
        pred6_blend = (1 - w) * rank(z6["pred"]) + w * rank(raw6)

        adj5 = weighted_auc(z5["events"], y5, pred5_blend)
        adj6 = weighted_auc(z6["events"], y6, pred6_blend)
        gain5 = adj5 - baseline_adj5
        gain6 = adj6 - baseline_adj6

        r = dict(
            blend_weight=w,
            day5_adjusted_auc=adj5,
            day5_gain=gain5,
            day6_adjusted_auc=adj6,
            day6_gain=gain6,
            both_pass=gain5 >= 0.0005 and gain6 >= 0.0005,
        )
        results.append(r)
        status = "PASS" if r["both_pass"] else "fail"
        print(
            f"  w={w:.2f}: day5 gain={gain5:+.6f}, day6 gain={gain6:+.6f} [{status}]"
        )

    best = max(results, key=lambda r: r["day5_gain"] + r["day6_gain"])
    elapsed = time.monotonic() - start

    report = dict(
        experiment="exp_c01_cooccurrence",
        hypothesis="Full-prefix item-item co-occurrence graph features capture broader "
                   "item similarity patterns than V53's 20-item ALS factors.",
        standalone_auc_day5=standalone_auc5,
        standalone_auc_day6=standalone_auc6,
        baseline_adjusted_auc_day5=baseline_adj5,
        baseline_adjusted_auc_day6=baseline_adj6,
        blend_results=results,
        best_blend=best,
        elapsed_seconds=elapsed,
        decision="KEEP" if best["both_pass"] else "REJECTED (below +0.0005 threshold)",
        params=params,
    )

    out_dir = ROOT / "track_c"
    (out_dir / "exp_c01_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / "exp_c01_predictions.npz",
        day5_oof=raw5_oof,
        day6_pred=raw6,
        day5_y=y5,
        day6_y=y6,
        day5_events=z5["events"],
        day6_events=z6["events"],
    )

    print(f"\n{'='*60}")
    print(f"RESULT: {report['decision']}")
    print(f"Best blend w={best['blend_weight']:.2f}: "
          f"day5={best['day5_gain']:+.6f}, day6={best['day6_gain']:+.6f}")
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"Report saved to track_c/exp_c01_report.json")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
