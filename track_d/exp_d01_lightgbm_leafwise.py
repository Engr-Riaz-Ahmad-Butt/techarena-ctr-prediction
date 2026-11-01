"""Exp D01: High-capacity leafwise LightGBM on V53 base features (190 features).

Tests different num_leaves (31, 63), learning rates, and regularization.
Evaluates both standalone and blended with V53 baseline using evaluate_candidate.py.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
TRACK_D = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_candidate import evaluate

def run_experiment(num_leaves=31, learning_rate=0.04, min_child_samples=100,
                   colsample_bytree=0.8, subsample=0.8, n_estimators=400,
                   seed=42, use_slate=False):
    exp_name = f"lgb_leaves{num_leaves}_lr{learning_rate}_n{n_estimators}_slate{int(use_slate)}_s{seed}"
    print(f"\n==========================================")
    print(f"Running Experiment: {exp_name}")
    print(f"==========================================")
    t0 = time.time()
    
    data = np.load(TRACK_D / "cache_data.npz")
    x5 = data["x5_slate"] if use_slate else data["x5_base"]
    x6 = data["x6_slate"] if use_slate else data["x6_base"]
    y5 = data["y5"]
    y6 = data["y6"]
    w5 = data["w5"]
    w6 = data["w6"]
    fold0_mask = data["fold0_mask"]
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "feature_fraction": colsample_bytree,
        "bagging_fraction": subsample,
        "bagging_freq": 1,
        "random_state": seed,
        "n_jobs": 16,
        "verbose": -1,
    }
    
    oof5 = np.zeros(len(y5), dtype=np.float64)
    
    # Fold 0: Train on fold 0, predict on fold 1
    print("Training Fold 0 (train on ~176k, predict on ~176k)...")
    dtrain0 = lgb.Dataset(x5[fold0_mask], label=y5[fold0_mask], weight=w5[fold0_mask])
    model0 = lgb.train(params, dtrain0, num_boost_round=n_estimators)
    oof5[~fold0_mask] = model0.predict(x5[~fold0_mask])
    
    # Fold 1: Train on fold 1, predict on fold 0
    print("Training Fold 1 (train on ~176k, predict on ~176k)...")
    dtrain1 = lgb.Dataset(x5[~fold0_mask], label=y5[~fold0_mask], weight=w5[~fold0_mask])
    model1 = lgb.train(params, dtrain1, num_boost_round=n_estimators)
    oof5[fold0_mask] = model1.predict(x5[fold0_mask])
    
    # Full fit on Day 5 -> predict on full Day 6
    print("Training Full Day 5 model (352k rows) -> predicting Day 6 (351k rows)...")
    dtrain_full = lgb.Dataset(x5, label=y5, weight=w5)
    model_full = lgb.train(params, dtrain_full, num_boost_round=n_estimators)
    pred6 = model_full.predict(x6)
    
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s.")
    
    # Standalone evaluation
    standalone_rep = evaluate(oof5, pred6, blend_weight=None, model_name=f"{exp_name}_standalone")
    print("\n--- STANDALONE PERFORMANCE ---")
    print(f"Day 5 Adj AUC: {standalone_rep['selection_day5']['adjusted_auc']:.6f} (gain: {standalone_rep['selection_day5']['gain']:+.6f})")
    print(f"Day 6 Adj AUC: {standalone_rep['selection_day6']['adjusted_auc']:.6f} (gain: {standalone_rep['selection_day6']['gain']:+.6f})")
    
    # Scan blend weights with V53
    best_blend = None
    best_gain_sum = -999.0
    blend_reports = {}
    
    weights_to_test = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
    print("\n--- BLENDING WITH V53 BASELINE ---")
    for bw in weights_to_test:
        rep = evaluate(oof5, pred6, blend_weight=bw, model_name=f"{exp_name}_blend_{bw}")
        g5 = rep["selection_day5"]["gain"]
        g6 = rep["selection_day6"]["gain"]
        passed = rep["selection_gate_passed"]
        holdout_info = ""
        if rep["fresh_holdout"]:
            gh = rep["fresh_holdout"]["gain"]
            holdout_info = f" | Fresh Holdout Gain: {gh:+.6f} ({'PASSED' if rep['accepted'] else 'FAILED'})"
        print(f"Blend Weight {bw:4.2f}: Day5 Gain = {g5:+.6f}, Day6 Gain = {g6:+.6f} | Gate: {'PASS' if passed else 'FAIL'}{holdout_info}")
        blend_reports[str(bw)] = rep
        if (g5 + g6) > best_gain_sum:
            best_gain_sum = g5 + g6
            best_blend = rep
            
    summary = {
        "exp_name": exp_name,
        "params": params,
        "n_estimators": n_estimators,
        "use_slate": use_slate,
        "runtime_seconds": round(elapsed, 2),
        "standalone": standalone_rep,
        "best_blend": best_blend,
        "blend_reports": blend_reports,
    }
    
    out_file = TRACK_D / f"{exp_name}_report.json"
    out_file.write_text(json.dumps(summary, indent=2))
    
    # Save predictions
    np.savez_compressed(
        TRACK_D / f"{exp_name}_preds.npz",
        oof5=oof5.astype(np.float32),
        pred6=pred6.astype(np.float32),
    )
    return summary

if __name__ == "__main__":
    # Test 1: baseline leafwise LightGBM (31 leaves, 400 rounds)
    run_experiment(num_leaves=31, learning_rate=0.04, n_estimators=400, use_slate=False)
