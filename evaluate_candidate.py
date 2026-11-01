"""Shared evaluation and gatekeeper module for all method tracks.

Checks candidate predictions on Selection Day 5 and Selection Day 6 against V53.
If and only if both selection days improve by >= +0.0005, evaluates on the
fresh untouched holdout. Stops and raises alert if single-step gain > +0.01.
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# Load frozen validation cohorts
SEL5 = np.load(ARTIFACTS / "selection_day5.npz")
SEL6 = np.load(ARTIFACTS / "selection_day6.npz")
HOLDOUT = np.load(ARTIFACTS / "fresh_holdout_day6.npz")


def rank(x):
    """Normalized average rank in [0, 1]."""
    x = np.asarray(x, dtype="float64")
    return rankdata(x, method="average") / (len(x) + 1)


def evaluate(day5_pred, day6_pred_full_or_sel, blend_weight=None, model_name="candidate"):
    """
    Evaluates predictions.
    day5_pred: array of 352,300 predictions for Selection Day 5
    day6_pred_full_or_sel: array of 351,380 (full Day 6) or 244,220 (selection Day 6)
    blend_weight: optional float in [0, 1] to blend candidate with V53 baseline
    """
    # Slice Day 6 if provided as full Day 6 (351,380)
    day6_arr = np.asarray(day6_pred_full_or_sel)
    if len(day6_arr) == 351380:
        sel6_pred = day6_arr[SEL6["full_indices"]]
        holdout_pred = day6_arr[HOLDOUT["full_indices"]]
    elif len(day6_arr) == len(SEL6["y"]):
        sel6_pred = day6_arr
        holdout_pred = None
    else:
        raise ValueError(f"Unexpected Day 6 prediction length: {len(day6_arr)}")

    # Handle rank blending if specified
    if blend_weight is not None and 0.0 < blend_weight <= 1.0:
        pred5 = (1.0 - blend_weight) * rank(SEL5["baseline"]) + blend_weight * rank(day5_pred)
        pred6 = (1.0 - blend_weight) * rank(SEL6["baseline"]) + blend_weight * rank(sel6_pred)
        pred_holdout = (
            (1.0 - blend_weight) * rank(HOLDOUT["baseline"]) + blend_weight * rank(holdout_pred)
            if holdout_pred is not None
            else None
        )
    else:
        pred5 = np.asarray(day5_pred)
        pred6 = sel6_pred
        pred_holdout = holdout_pred

    # Selection Day 5 metrics
    ord5 = float(roc_auc_score(SEL5["y"], pred5))
    adj5 = float(roc_auc_score(SEL5["y"], pred5, sample_weight=SEL5["weights"]))
    base_adj5 = float(roc_auc_score(SEL5["y"], SEL5["baseline"], sample_weight=SEL5["weights"]))
    gain5 = adj5 - base_adj5

    # Selection Day 6 metrics
    ord6 = float(roc_auc_score(SEL6["y"], pred6))
    adj6 = float(roc_auc_score(SEL6["y"], pred6, sample_weight=SEL6["weights"]))
    base_adj6 = float(roc_auc_score(SEL6["y"], SEL6["baseline"], sample_weight=SEL6["weights"]))
    gain6 = adj6 - base_adj6

    # Leakage check: gain > 0.01 triggers immediate stop
    if gain5 > 0.01 or gain6 > 0.01:
        raise ValueError(
            f"LEAKAGE ALERT! Single step gain exceeded +0.01: "
            f"day5_gain={gain5:+.6f}, day6_gain={gain6:+.6f}. "
            f"Immediate audit required."
        )

    # Acceptance gate: gain >= +0.0005 on BOTH selection days
    both_selection_passed = (gain5 >= 0.0005) and (gain6 >= 0.0005)

    # Fresh holdout evaluation (only queried if selection passed and full predictions supplied)
    holdout_metrics = None
    if both_selection_passed and pred_holdout is not None:
        ord_h = float(roc_auc_score(HOLDOUT["y"], pred_holdout))
        adj_h = float(roc_auc_score(HOLDOUT["y"], pred_holdout, sample_weight=HOLDOUT["weights"]))
        base_adj_h = float(
            roc_auc_score(HOLDOUT["y"], HOLDOUT["baseline"], sample_weight=HOLDOUT["weights"])
        )
        gain_h = adj_h - base_adj_h
        holdout_metrics = {
            "ordinary_auc": ord_h,
            "adjusted_auc": adj_h,
            "baseline_adjusted_auc": base_adj_h,
            "gain": gain_h,
            "holdout_passed": gain_h >= 0.0,
        }

    report = {
        "model_name": model_name,
        "blend_weight": blend_weight,
        "selection_day5": {
            "ordinary_auc": ord5,
            "adjusted_auc": adj5,
            "baseline_adjusted_auc": base_adj5,
            "gain": gain5,
        },
        "selection_day6": {
            "ordinary_auc": ord6,
            "adjusted_auc": adj6,
            "baseline_adjusted_auc": base_adj6,
            "gain": gain6,
        },
        "selection_gate_passed": both_selection_passed,
        "fresh_holdout": holdout_metrics,
        "accepted": (
            both_selection_passed
            and (holdout_metrics is not None and holdout_metrics["holdout_passed"])
        ),
    }
    return report


if __name__ == "__main__":
    # Self-test using V53 baseline itself (gain should be 0.000000)
    rep = evaluate(SEL5["baseline"], SEL6["baseline"], model_name="v53_baseline_self_test")
    print("Self-test report:", rep)
    assert abs(rep["selection_day5"]["gain"]) < 1e-9
    assert abs(rep["selection_day6"]["gain"]) < 1e-9
    print("Self-test passed!")
