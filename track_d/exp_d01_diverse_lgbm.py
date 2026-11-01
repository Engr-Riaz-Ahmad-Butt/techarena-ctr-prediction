"""Track D Experiment 01: Diverse multi-seed LightGBM on V53 features + new slate features.

Hypothesis: A leafwise LightGBM trained on the same causal features as V53 (plus
new slate-relative features) provides complementary signal because it uses a different
splitting strategy (leafwise vs CatBoost's depth-wise). Multi-seed averaging reduces
variance. Training on days 2-4 (weekdays) should align better with the evaluation
period which likely contains weekday events.

New features added on top of V53 baseline logit:
  1. Slate rank: candidate rank within event by V53 prediction
  2. Slate gap: candidate score minus event max score
  3. Slate spread: standard deviation of scores within event
  4. User activity level: log of total events before cutoff
  5. Item recency: days since last click on this item by anyone
  6. V53 baseline logit

Model: LightGBM with 3 different seeds, averaged.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def rank(x):
    return rankdata(np.asarray(x, dtype="float64"), method="average") / (len(x) + 1)


def build_features(train, hist, hist_times, v53_pred, events_10x, cutoff_ts):
    """Build V53 logit + new slate-relative + user/item temporal features."""
    n_rows = len(v53_pred)
    n_events = n_rows // 10

    # V53 baseline logit
    v53_logit = logit(np.clip(v53_pred, 1e-6, 1 - 1e-6))

    # Slate-relative features from V53 predictions
    pred_reshaped = v53_pred.reshape(n_events, 10)
    slate_mean = np.repeat(pred_reshaped.mean(axis=1), 10)
    slate_max = np.repeat(pred_reshaped.max(axis=1), 10)
    slate_std = np.repeat(pred_reshaped.std(axis=1), 10)
    slate_rank = np.zeros(n_rows)
    for i in range(n_events):
        block = v53_pred[i*10:(i+1)*10]
        slate_rank[i*10:(i+1)*10] = rankdata(-block) / 11.0  # higher pred = lower rank

    slate_gap = v53_pred - slate_max
    slate_diff = v53_pred - slate_mean

    # User activity features
    prior = np.asarray(train[train[:, 2] < cutoff_ts])
    prior_events = prior[::10]
    user_event_counts = np.bincount(prior_events[:, 0], minlength=int(train[:, 0].max()) + 1)
    user_click_counts = np.bincount(
        prior_events[:, 0],
        weights=prior[:, 5].reshape(-1, 10).sum(1),
        minlength=int(train[:, 0].max()) + 1,
    )

    # Item temporal features
    item_last_click_ts = np.full(int(train[:, 1].max()) + 1, -1, dtype="int64")
    item_exposure_count = np.bincount(prior[:, 1], minlength=int(train[:, 1].max()) + 1)
    item_click_count = np.bincount(
        prior[:, 1], weights=prior[:, 5], minlength=int(train[:, 1].max()) + 1
    )
    for r in range(len(prior)):
        if int(prior[r, 5]) == 1:
            iid = int(prior[r, 1])
            ts = int(prior[r, 2])
            if ts > item_last_click_ts[iid]:
                item_last_click_ts[iid] = ts

    # Map to validation rows
    rows = np.asarray(train[events_10x])
    users = rows[:, 0]
    items = rows[:, 1]

    user_events_feat = np.log1p(user_event_counts[users])
    user_clicks_feat = np.log1p(user_click_counts[users])
    user_ctr_feat = user_click_counts[users] / np.maximum(user_event_counts[users], 1)

    item_exp_feat = np.log1p(item_exposure_count[items])
    item_click_feat = np.log1p(item_click_count[items])
    item_recency = np.where(
        item_last_click_ts[items] > 0,
        (cutoff_ts - item_last_click_ts[items]) / 86400.0,
        30.0,  # never clicked = 30 days
    )

    # History length
    event_indices = events_10x[::10] // 10
    h_lengths = (hist[event_indices] >= 0).sum(axis=1)
    history_len = np.repeat(h_lengths, 10)

    # History recency (mean age in days)
    h_ages = np.zeros(len(event_indices), dtype="float64")
    for i, ev_idx in enumerate(event_indices):
        ev_ts = int(train[ev_idx * 10, 2])
        valid = hist_times[ev_idx][hist[ev_idx] >= 0]
        if len(valid) > 0:
            h_ages[i] = np.mean((ev_ts - valid) / 86400.0)
    history_mean_age = np.repeat(h_ages, 10)

    X = np.column_stack([
        v53_logit,
        slate_rank,
        slate_gap,
        slate_diff,
        slate_std,
        user_events_feat,
        user_clicks_feat,
        user_ctr_feat,
        item_exp_feat,
        item_click_feat,
        item_recency,
        history_len,
        history_mean_age,
    ])

    feature_names = [
        "v53_logit", "slate_rank", "slate_gap", "slate_diff", "slate_std",
        "user_events", "user_clicks", "user_ctr",
        "item_exposures", "item_clicks", "item_recency",
        "history_len", "history_mean_age",
    ]

    assert np.isfinite(X).all(), "Non-finite features detected!"
    return X, feature_names


def main():
    start = time.monotonic()
    print("=" * 60)
    print("Track D Experiment 01: Diverse multi-seed LightGBM")
    print("=" * 60)

    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    hist = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    hist_times = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")
    z5 = np.load(ARTIFACTS / "v53_causal_propensity_day5.npz")
    z6 = np.load(ARTIFACTS / "v53_causal_propensity_day6.npz")

    START_TS = int(train[0, 2])
    DAY = 86400
    cutoff5 = START_TS + 5 * DAY
    cutoff6 = START_TS + 6 * DAY

    # Build features
    print(f"\n[{time.monotonic()-start:.0f}s] Building features for Day 5...")
    ev5_10x = z5["events"]  # row indices (10 per event)
    X5, feat_names = build_features(train, hist, hist_times, z5["pred"], ev5_10x, cutoff5)
    y5 = z5["y"].copy()
    print(f"  Shape: {X5.shape}, Features: {feat_names}")

    print(f"[{time.monotonic()-start:.0f}s] Building features for Day 6...")
    ev6_10x = z6["events"]
    X6, _ = build_features(train, hist, hist_times, z6["pred"], ev6_10x, cutoff6)
    y6 = z6["y"].copy()
    print(f"  Shape: {X6.shape}")

    # Evaluation weights
    eh = np.load(ARTIFACTS / "evaluation_hist.npy", mmap_mode="r")
    h_full = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)

    def weighted_auc(events, y, pred):
        vg = np.digitize((h_full[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
        w = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
        return float(roc_auc_score(y, pred, sample_weight=w))

    baseline_adj5 = weighted_auc(z5["events"], y5, z5["pred"])
    baseline_adj6 = weighted_auc(z6["events"], y6, z6["pred"])
    print(f"\nBaseline V53 adjusted AUC: day5={baseline_adj5:.6f}, day6={baseline_adj6:.6f}")

    # Multi-seed LightGBM
    seeds = [7101, 7102, 7103]
    lgb_params = dict(
        objective="binary",
        metric="auc",
        num_leaves=63,
        learning_rate=0.03,
        n_estimators=500,
        min_child_samples=200,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=2,
    )

    # Train on Day 5, predict Day 6 (multi-seed)
    raw6_seeds = []
    for seed in seeds:
        print(f"\n[{time.monotonic()-start:.0f}s] Training LightGBM seed={seed} on Day 5...")
        model = lgb.LGBMClassifier(random_state=seed, **lgb_params)
        model.fit(X5, y5)
        p6 = model.predict_proba(X6)[:, 1]
        raw6_seeds.append(p6)
        auc6 = roc_auc_score(y6, p6)
        print(f"  Seed {seed} standalone Day6 AUC: {auc6:.6f}")

    raw6_mean = np.mean(raw6_seeds, axis=0)
    print(f"\n  3-seed average Day6 AUC: {roc_auc_score(y6, raw6_mean):.6f}")

    # Cross-fit Day 5 OOF (2 folds, multi-seed)
    print(f"\n[{time.monotonic()-start:.0f}s] Cross-fitting Day 5...")
    users5 = train[z5["events"], 0]
    unique_users5 = np.unique(users5)
    rng = np.random.default_rng(7104)
    rng.shuffle(unique_users5)
    mid = len(unique_users5) // 2
    fold_a_users = set(unique_users5[:mid].tolist())
    fold_a_mask = np.array([int(u) in fold_a_users for u in users5])
    fold_b_mask = ~fold_a_mask

    raw5_seeds = [np.zeros(len(y5), dtype="float64") for _ in seeds]
    for s_idx, seed in enumerate(seeds):
        for fold_name, fit_mask, pred_mask in [("A", fold_a_mask, fold_b_mask), ("B", fold_b_mask, fold_a_mask)]:
            m = lgb.LGBMClassifier(random_state=seed, **lgb_params)
            m.fit(X5[fit_mask], y5[fit_mask])
            raw5_seeds[s_idx][pred_mask] = m.predict_proba(X5[pred_mask])[:, 1]

    raw5_mean = np.mean(raw5_seeds, axis=0)
    print(f"  3-seed average Day5 OOF AUC: {roc_auc_score(y5, raw5_mean):.6f}")

    # Evaluate blends
    print(f"\n[{time.monotonic()-start:.0f}s] Evaluating rank blends with V53...")
    results = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred5 = (1 - w) * rank(z5["pred"]) + w * rank(raw5_mean)
        pred6 = (1 - w) * rank(z6["pred"]) + w * rank(raw6_mean)

        adj5 = weighted_auc(z5["events"], y5, pred5)
        adj6 = weighted_auc(z6["events"], y6, pred6)
        gain5 = adj5 - baseline_adj5
        gain6 = adj6 - baseline_adj6

        r = dict(
            blend_weight=w, day5_adjusted_auc=adj5, day5_gain=gain5,
            day6_adjusted_auc=adj6, day6_gain=gain6,
            both_pass=gain5 >= 0.0005 and gain6 >= 0.0005,
        )
        results.append(r)
        status = "PASS" if r["both_pass"] else "fail"
        print(f"  w={w:.2f}: day5 gain={gain5:+.6f}, day6 gain={gain6:+.6f} [{status}]")

    best = max(results, key=lambda r: r["day5_gain"] + r["day6_gain"])
    elapsed = time.monotonic() - start

    report = dict(
        experiment="exp_d01_diverse_lgbm",
        hypothesis="Leafwise LightGBM + slate-relative features provides complementary signal to CatBoost V53.",
        standalone_auc_day5=float(roc_auc_score(y5, raw5_mean)),
        standalone_auc_day6=float(roc_auc_score(y6, raw6_mean)),
        baseline_adjusted_auc_day5=baseline_adj5,
        baseline_adjusted_auc_day6=baseline_adj6,
        blend_results=results,
        best_blend=best,
        elapsed_seconds=elapsed,
        decision="KEEP" if best["both_pass"] else "REJECTED (below +0.0005 threshold)",
        lgb_params=lgb_params,
        seeds=seeds,
        feature_names=feat_names,
    )

    out_dir = ROOT / "track_d"
    (out_dir / "exp_d01_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / "exp_d01_predictions.npz",
        day5_oof=raw5_mean,
        day6_pred=raw6_mean,
        day5_y=y5, day6_y=y6,
        day5_events=z5["events"], day6_events=z6["events"],
    )

    print(f"\n{'='*60}")
    print(f"RESULT: {report['decision']}")
    print(f"Best blend w={best['blend_weight']:.2f}: "
          f"day5={best['day5_gain']:+.6f}, day6={best['day6_gain']:+.6f}")
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
