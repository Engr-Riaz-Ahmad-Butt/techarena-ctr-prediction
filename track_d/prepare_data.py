"""Extract and cache V53 features and build slate-relative features for Track D."""
import os
import sys
import pickle
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACK_D = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def compute_intra_slate_features(scores_1d):
    """
    Given a 1D array of scores (length = num_events * 10),
    compute:
    - diff_mean: score - slate_mean
    - diff_max: score - slate_max
    - diff_min: score - slate_min
    - intra_rank: normalized rank within slate [0, 1]
    - softmax_1: softmax(s) within slate
    - softmax_05: softmax(s / 0.5) within slate
    - softmax_2: softmax(s / 2.0) within slate
    - is_top: whether candidate has max score in slate
    - margin_to_top2: for top candidate, diff to 2nd; for others, diff to top
    """
    scores = np.asarray(scores_1d, dtype=np.float32).reshape(-1, 10)
    num_events = len(scores)
    
    # mean, max, min
    s_mean = scores.mean(axis=1, keepdims=True)
    s_max = scores.max(axis=1, keepdims=True)
    s_min = scores.min(axis=1, keepdims=True)
    s_std = scores.std(axis=1, keepdims=True) + 1e-6
    
    diff_mean = (scores - s_mean).ravel()
    diff_max = (scores - s_max).ravel()
    diff_min = (scores - s_min).ravel()
    z_score = ((scores - s_mean) / s_std).ravel()
    
    # intra-slate rank (0.0 to 1.0, 1.0 being highest)
    # argsort gives rank in ascending order
    order = np.argsort(scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    np.put_along_axis(ranks, order, np.tile(np.arange(10, dtype=np.float32) / 9.0, (num_events, 1)), axis=1)
    intra_rank = ranks.ravel()
    
    # softmax at various temperatures
    def smax(temp):
        shifted = (scores - s_max) / temp
        exp = np.exp(np.clip(shifted, -20.0, 0.0))
        denom = exp.sum(axis=1, keepdims=True) + 1e-12
        return (exp / denom).ravel()
    
    sm_1 = smax(1.0)
    sm_05 = smax(0.5)
    sm_2 = smax(2.0)
    
    # is_top
    is_top = (ranks == 1.0).astype(np.float32).ravel()
    
    # margin to 2nd:
    # 2nd highest score in each event
    sorted_scores = np.sort(scores, axis=1)
    second_best = sorted_scores[:, -2:-1]  # shape (num_events, 1)
    # for top candidate: score - second_best; for non-top candidate: score - top
    is_top_2d = (ranks == 1.0)
    margin = np.where(is_top_2d, scores - second_best, scores - s_max).astype(np.float32).ravel()
    
    return np.column_stack([
        diff_mean, diff_max, diff_min, z_score,
        intra_rank, sm_1, sm_05, sm_2, is_top, margin
    ])

def prepare_all():
    start_time = time.time()
    print("Preparing data for Track D...")
    
    with (ARTIFACTS / "v48_history_calibrator.pkl").open("rb") as f:
        cal = pickle.load(f)
        spline = cal["spline"]
        fit_users = np.asarray(cal["fit_users"])
        
    t = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    h = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    ht = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")
    
    sel5 = np.load(ARTIFACTS / "selection_day5.npz")
    sel6 = np.load(ARTIFACTS / "selection_day6.npz")
    holdout = np.load(ARTIFACTS / "fresh_holdout_day6.npz")
    v53_5 = np.load(ARTIFACTS / "v53_causal_propensity_day5.npz")
    v53_6 = np.load(ARTIFACTS / "v53_causal_propensity_day6.npz")
    
    from v53_causal_propensity import features as v53_features
    
    print("Extracting Day 5 features...")
    z5, x5_base, u5, g5, names5 = v53_features(5, t, h, ht, spline)
    print(f"Day 5 raw features extracted: {x5_base.shape}")
    
    print("Extracting Day 6 features...")
    z6, x6_base, u6, g6, names6 = v53_features(6, t, h, ht, spline)
    print(f"Day 6 raw features extracted: {x6_base.shape}")
    
    y5 = np.asarray(v53_5["y"], dtype=np.float32)
    y6 = np.asarray(v53_6["y"], dtype=np.float32)
    events5 = np.asarray(v53_5["events"], dtype=np.int32)
    events6 = np.asarray(v53_6["events"], dtype=np.int32)
    
    v53_pred5 = np.asarray(v53_5["pred"], dtype=np.float32)
    v53_pred6 = np.asarray(v53_6["pred"], dtype=np.float32)
    
    w5 = np.asarray(sel5["weights"], dtype=np.float32)
    w6 = np.zeros(len(y6), dtype=np.float32)
    w6[sel6["full_indices"]] = sel6["weights"]
    w6[holdout["full_indices"]] = holdout["weights"]
    
    # User partition for Day 5 (Fold 0 vs Fold 1)
    fold0_mask = np.isin(u5, fit_users)
    print(f"Day 5 user split: Fold 0 rows={fold0_mask.sum()}, Fold 1 rows={(~fold0_mask).sum()}")
    
    # Compute slate features for V53 prediction
    print("Computing slate features on V53 predictions...")
    slate_v53_5 = compute_intra_slate_features(v53_pred5)
    slate_v53_6 = compute_intra_slate_features(v53_pred6)
    
    # Compute slate features for ALS score z['pred']
    print("Computing slate features on ALS baseline scores...")
    slate_als_5 = compute_intra_slate_features(z5["pred"])
    slate_als_6 = compute_intra_slate_features(z6["pred"])
    
    # Compute slate features for calibrated score
    print("Computing slate features on V53 calibrated scores...")
    slate_cal_5 = compute_intra_slate_features(v53_5["calibrated"])
    slate_cal_6 = compute_intra_slate_features(v53_6["calibrated"])
    
    # Cross-feature interactions:
    # Look at x5_base features:
    # 0: spline bases, 62.. base
    # In v53_features, added features are appended:
    # Let's compute targeted non-linear interactions:
    # User propensity (column -22 approx) and item CTR (column -18 approx)
    # We can also compute candidate position in slate:
    pos_in_slate = np.tile(np.arange(10, dtype=np.float32), len(y5) // 10)[:, None]
    pos_in_slate_6 = np.tile(np.arange(10, dtype=np.float32), len(y6) // 10)[:, None]
    
    # Combine:
    # Feature set A: raw v53 features (190 columns)
    # Feature set B: raw v53 + V53 slate features (190 + 10 = 200)
    # Feature set C: raw v53 + all slate features (190 + 10 + 10 + 10 + 1 = 221)
    x5_slate = np.column_stack([
        x5_base.astype(np.float32),
        slate_v53_5,
        slate_als_5,
        slate_cal_5,
        pos_in_slate
    ])
    x6_slate = np.column_stack([
        x6_base.astype(np.float32),
        slate_v53_6,
        slate_als_6,
        slate_cal_6,
        pos_in_slate_6
    ])
    print(f"Combined features shape: {x5_slate.shape}")
    
    # Save cache
    cache_path = TRACK_D / "cache_data.npz"
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(
        cache_path,
        x5_base=x5_base.astype(np.float32),
        x6_base=x6_base.astype(np.float32),
        x5_slate=x5_slate.astype(np.float32),
        x6_slate=x6_slate.astype(np.float32),
        y5=y5,
        y6=y6,
        events5=events5,
        events6=events6,
        w5=w5,
        w6=w6,
        fold0_mask=fold0_mask,
        v53_pred5=v53_pred5,
        v53_pred6=v53_pred6,
    )
    print(f"Data preparation complete in {round(time.time() - start_time, 2)}s. File size: {os.path.getsize(cache_path) / (1024**2):.1f} MB")

if __name__ == "__main__":
    prepare_all()
