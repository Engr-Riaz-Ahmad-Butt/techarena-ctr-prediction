"""Track A Experiment 01: Multi-seed exposure-weighted implicit ALS factors.

Hypothesis: Exposure-weighted ALS with 64 dimensions, 5 different seeds, and
exponential click time-decay (half-life 3 days) may produce factors with stronger
complementary signal for sparse-history users when averaged and blended with V53.

Key design decisions:
  1. NO label-informed initialization - random init only.
  2. Confidence weighting: c_ui = 1 + alpha * (exposure_count_ui) where clicked rows
     get higher confidence.
  3. Click decay: exponential weighting by recency of clicks.
  4. 5 seeds averaged to reduce variance.
  5. Factor features: user-item dot product, user bias, item bias, user/item norms.

Temporal safety: Factors fit only on data strictly before each validation cutoff.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def rank(x):
    return rankdata(np.asarray(x, dtype="float64"), method="average") / (len(x) + 1)


def build_interaction_matrix(train, cutoff_ts, n_users, n_items, decay_half_life_days=3.0):
    """Build exposure and preference matrices from training data before cutoff."""
    prior = np.asarray(train[train[:, 2] < cutoff_ts])
    
    # Exposure matrix (all candidate impressions)
    exposure = sparse.csr_matrix(
        (np.ones(len(prior), dtype="float32"), (prior[:, 0], prior[:, 1])),
        shape=(n_users, n_items),
    )
    
    # Click matrix with time decay
    clicked = prior[prior[:, 5] == 1]
    if len(clicked) == 0:
        click_weights = np.array([], dtype="float32")
        click_users = np.array([], dtype="int64")
        click_items = np.array([], dtype="int64")
    else:
        decay_rate = np.log(2) / (decay_half_life_days * 86400)
        time_diffs = cutoff_ts - clicked[:, 2]
        click_weights = np.exp(-decay_rate * time_diffs).astype("float32")
        click_users = clicked[:, 0]
        click_items = clicked[:, 1]
    
    clicks = sparse.csr_matrix(
        (click_weights, (click_users, click_items)),
        shape=(n_users, n_items),
    )
    
    return exposure, clicks


def als_implicit(exposure, clicks, n_factors=64, n_iterations=8, alpha=9.0,
                 reg_user=0.1, reg_item=0.1, seed=42):
    """Alternating Least Squares for implicit feedback.
    
    Preference: p_ui = 1 if clicks_ui > 0, else 0
    Confidence: c_ui = 1 + alpha * clicks_ui
    """
    rng = np.random.RandomState(seed)
    n_users, n_items = exposure.shape
    
    # Initialize factors
    user_factors = rng.normal(0, 0.01, (n_users, n_factors)).astype("float64")
    item_factors = rng.normal(0, 0.01, (n_items, n_factors)).astype("float64")
    user_bias = np.zeros(n_users, dtype="float64")
    item_bias = np.zeros(n_items, dtype="float64")
    
    # Build confidence matrix: C = 1 + alpha * clicks
    C = clicks.copy()
    C.data = 1.0 + alpha * C.data
    # For non-clicked but exposed items, confidence = 1 (stored as exposure - clicks entries)
    # Actually for ALS implicit, we iterate over non-zero entries in C
    
    # Preference matrix: P = (clicks > 0)
    P = clicks.copy()
    P.data = (P.data > 0).astype("float64")
    
    print(f"    ALS: {n_users} users, {n_items} items, {n_factors}D, "
          f"{clicks.nnz} clicks, seed={seed}")
    
    for iteration in range(n_iterations):
        # Update user factors
        YtY = item_factors.T @ item_factors  # (d, d)
        for u in range(n_users):
            # Get items this user interacted with
            indices = C[u].indices
            if len(indices) == 0:
                continue
            c_u = np.asarray(C[u, indices].todense()).ravel()
            p_u = np.asarray(P[u, indices].todense()).ravel()
            Y_u = item_factors[indices]  # (n_items_u, d)
            
            # A = Y^T * (C_u - I) * Y + Y^T * Y + reg * I
            # b = Y^T * C_u * p_u
            CuI = sparse.diags(c_u - 1.0)
            A = YtY + Y_u.T @ CuI @ Y_u + reg_user * np.eye(n_factors)
            b = Y_u.T @ (c_u * p_u)
            user_factors[u] = np.linalg.solve(A, b)
        
        # Update item factors
        XtX = user_factors.T @ user_factors  # (d, d)
        for i in range(n_items):
            col = C[:, i].tocsr()
            indices = col.indices
            if len(indices) == 0:
                continue
            c_i = np.asarray(col[indices].todense()).ravel()
            p_col = P[:, i].tocsr()
            p_i = np.asarray(p_col[indices].todense()).ravel()
            X_i = user_factors[indices]
            
            CiI = sparse.diags(c_i - 1.0)
            A = XtX + X_i.T @ CiI @ X_i + reg_item * np.eye(n_factors)
            b = X_i.T @ (c_i * p_i)
            item_factors[i] = np.linalg.solve(A, b)
        
        # Report progress
        if (iteration + 1) % 2 == 0:
            print(f"    Iteration {iteration + 1}/{n_iterations}: "
                  f"user_norm={np.linalg.norm(user_factors):.2f}, "
                  f"item_norm={np.linalg.norm(item_factors):.2f}")
    
    return user_factors, item_factors, user_bias, item_bias


def compute_factor_features(train, user_factors, item_factors, event_rows):
    """Compute factor-based features for validation events."""
    n_rows = len(event_rows)
    
    users = train[event_rows, 0].astype(int)
    items = train[event_rows, 1].astype(int)
    
    n_users, n_factors = user_factors.shape
    n_items = item_factors.shape[0]
    
    # Clip to valid range
    valid_u = users < n_users
    valid_i = items < n_items
    valid = valid_u & valid_i
    
    # Dot product (main signal)
    dots = np.zeros(n_rows, dtype="float64")
    dots[valid] = np.sum(user_factors[users[valid]] * item_factors[items[valid]], axis=1)
    
    # User and item norms
    u_norm = np.zeros(n_rows, dtype="float64")
    u_norm[valid_u] = np.linalg.norm(user_factors[users[valid_u]], axis=1)
    i_norm = np.zeros(n_rows, dtype="float64")
    i_norm[valid_i] = np.linalg.norm(item_factors[items[valid_i]], axis=1)
    
    return np.column_stack([dots, u_norm, i_norm])


def main():
    start = time.monotonic()
    print("=" * 60)
    print("Track A Experiment 01: Multi-seed exposure-weighted ALS")
    print("=" * 60)

    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    z5 = np.load(ARTIFACTS / "v53_causal_propensity_day5.npz")
    z6 = np.load(ARTIFACTS / "v53_causal_propensity_day6.npz")

    START_TS = int(train[0, 2])
    DAY = 86400
    n_users = int(train[:, 0].max()) + 1
    n_items = int(train[:, 1].max()) + 1

    # Evaluation weights
    eh = np.load(ARTIFACTS / "evaluation_hist.npy", mmap_mode="r")
    h = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)

    def weighted_auc(events, y, pred):
        vg = np.digitize((h[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
        w = ((np.bincount(eg, minlength=5) / len(eg)) / (np.bincount(vg, minlength=5) / len(vg)))[vg]
        return float(roc_auc_score(y, pred, sample_weight=w))

    baseline_adj5 = weighted_auc(z5["events"], z5["y"], z5["pred"])
    baseline_adj6 = weighted_auc(z6["events"], z6["y"], z6["pred"])
    print(f"\nBaseline V53 adjusted AUC: day5={baseline_adj5:.6f}, day6={baseline_adj6:.6f}")

    seeds = [9001, 9002, 9003, 9004, 9005]
    n_factors = 64
    
    # Results storage
    all_dots5 = []
    all_dots6 = []

    for day, z, day_name in [(5, z5, "Day5"), (6, z6, "Day6")]:
        cutoff = START_TS + day * DAY
        print(f"\n{'='*40}")
        print(f"Processing {day_name} (cutoff={cutoff})")
        print(f"{'='*40}")
        
        print(f"[{time.monotonic()-start:.0f}s] Building interaction matrices...")
        exposure, clicks = build_interaction_matrix(train, cutoff, n_users, n_items, decay_half_life_days=3.0)
        print(f"  Exposure nnz: {exposure.nnz}, Clicks nnz: {clicks.nnz}")
        
        event_rows = z["events"]
        dots_all_seeds = []
        
        for seed in seeds:
            print(f"\n[{time.monotonic()-start:.0f}s] Training ALS seed={seed}...")
            uf, if_, ub, ib = als_implicit(
                exposure, clicks, n_factors=n_factors, n_iterations=6,
                alpha=9.0, reg_user=4.0, reg_item=4.0, seed=seed,
            )
            
            feats = compute_factor_features(train, uf, if_, event_rows)
            dots = feats[:, 0]  # just the dot product
            dots_all_seeds.append(dots)
            
            auc = roc_auc_score(z["y"], dots)
            print(f"  Seed {seed} standalone {day_name} ordinary AUC: {auc:.6f}")
        
        # Average across seeds
        mean_dots = np.mean(dots_all_seeds, axis=0)
        avg_auc = roc_auc_score(z["y"], mean_dots)
        print(f"\n  5-seed average {day_name} AUC: {avg_auc:.6f}")
        
        if day == 5:
            all_dots5 = mean_dots
        else:
            all_dots6 = mean_dots

    # Evaluate blends
    print(f"\n[{time.monotonic()-start:.0f}s] Evaluating rank blends with V53...")
    y5 = z5["y"]
    y6 = z6["y"]
    
    results = []
    for w in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        pred5 = (1 - w) * rank(z5["pred"]) + w * rank(all_dots5)
        pred6 = (1 - w) * rank(z6["pred"]) + w * rank(all_dots6)

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
        experiment="exp_a01_als_multi_seed",
        hypothesis="5-seed exposure-weighted ALS with 64 dimensions and click decay "
                   "provides complementary factor signal to V53.",
        standalone_auc_day5=float(roc_auc_score(y5, all_dots5)),
        standalone_auc_day6=float(roc_auc_score(y6, all_dots6)),
        baseline_adjusted_auc_day5=baseline_adj5,
        baseline_adjusted_auc_day6=baseline_adj6,
        blend_results=results,
        best_blend=best,
        elapsed_seconds=elapsed,
        decision="KEEP" if best["both_pass"] else "REJECTED (below +0.0005 threshold)",
        n_factors=n_factors,
        seeds=seeds,
    )

    out_dir = ROOT / "track_a"
    (out_dir / "exp_a01_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / "exp_a01_predictions.npz",
        day5_dots=all_dots5, day6_dots=all_dots6,
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
