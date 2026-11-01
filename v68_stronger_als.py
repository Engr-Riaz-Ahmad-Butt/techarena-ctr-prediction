"""Exploratory stronger user-item ALS personalization pass.

This is development-only evidence on the reused day5/day6 first-event cohorts.
It is not a promotion candidate and must not be reported as a fresh holdout.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
from imp15_als import prepare, split, cutoff_of, score, augmented, solve_rows, nu, ni

ROOT = Path(__file__).resolve().parent
OUT = R / "v68_stronger_als"
DEFAULT_DIMS = [64, 128, 256]
DEFAULT_SEEDS = [101, 202, 303, 404, 505]
BETA_GRID = [0.05, 0.10, 0.20]


def _rank(values):
    return rankdata(values, method="average") / (len(values) + 1.0)


def _weighted_auc(y, pred, events):
    hist = np.load(R / "train_hist.npy", mmap_mode="r")
    eval_hist = np.load(R / "evaluation_hist.npy", mmap_mode="r")
    eval_groups = np.digitize((eval_hist >= 0).sum(axis=1), [0, 4, 9, 19], right=True)
    val_groups = np.digitize((hist[events] >= 0).sum(axis=1), [0, 4, 9, 19], right=True)
    target = np.bincount(eval_groups, minlength=5).astype(np.float64)
    target /= target.sum()
    source = np.bincount(val_groups, minlength=5).astype(np.float64)
    source /= source.sum()
    sample_weights = np.divide(target, source, out=np.zeros(5), where=source > 0)[val_groups]
    sample_weights /= sample_weights.mean()
    return float(roc_auc_score(y, pred, sample_weight=sample_weights))


def _initial_v(dim, day, seed):
    base = np.load(R / f"imp15_model_day{day}.npz")
    v = base["V"].astype(np.float64)
    if v.shape[1] < dim:
        rng = np.random.default_rng(seed)
        extra = rng.normal(0.0, 0.05, size=(v.shape[0], dim - v.shape[1]))
        v = np.concatenate([v, extra], axis=1)
    else:
        v = v[:, :dim].copy()
    return v


def _fit_single(day, dim, seed, iterations=6):
    started = time.monotonic()
    w, t, wt, tt, g, up, ip, confidence, exposures = prepare(day)
    v = _initial_v(dim, day, seed)
    u = np.zeros((nu, dim), dtype=np.float64)
    bu = up.copy()
    bi = ip.copy()

    for iteration in range(1, iterations + 1):
        x = augmented(v)
        bu, u = solve_rows(
            w.indptr,
            w.indices,
            w.data,
            t.data,
            v,
            bi,
            x.T @ x,
            -x.T @ (g + bi),
            confidence,
            g,
            up,
            latent_reg=4.0,
        )
        x = augmented(u)
        bi, v = solve_rows(
            wt.indptr,
            wt.indices,
            wt.data,
            tt.data,
            u,
            bu,
            x.T @ (confidence[:, None] * x),
            -x.T @ (confidence * (g + bu)),
            np.ones(ni),
            g,
            ip,
            latent_reg=4.0,
        )

    _, val = split(day)
    ref = np.load(R / f"v53_causal_propensity_day{day}.npz")
    if not (np.array_equal(val[:, 6], ref["events"]) and np.array_equal(val[:, 5], ref["y"])):
        raise AssertionError(f"Validation rows do not match saved V53 cohort for day {day}")
    vals = ref["y"]
    baseline = ref["pred"]
    baseline_rank = _rank(baseline)
    raw = score(val, u, v, bu, bi, g, exposures)
    raw_rank = _rank(raw)
    best = None
    trials = []
    for weight in BETA_GRID:
        pred = (1.0 - weight) * baseline_rank + weight * raw_rank
        pred = np.asarray(pred, dtype=np.float64)
        auc = float(roc_auc_score(vals, pred))
        weighted_auc = _weighted_auc(vals, pred, ref["events"])
        delta = weighted_auc - _weighted_auc(vals, baseline, ref["events"])
        trials.append(
            {
                "day": day,
                "dim": dim,
                "seed": seed,
                "weight": weight,
                "auc": auc,
                "weighted_auc": weighted_auc,
                "gain_vs_v53": delta,
            }
        )
        if best is None or weighted_auc > best["weighted_auc"]:
            best = {
                "day": day,
                "dim": dim,
                "seed": seed,
                "weight": weight,
                "auc": auc,
                "weighted_auc": weighted_auc,
                "gain_vs_v53": delta,
                "raw_pred": raw,
                "blend_pred": pred,
            }

    result = {
        "exploratory_only": True,
        "status": "development_only_reused_cohorts",
        "day": day,
        "dim": dim,
        "seed": seed,
        "iterations": iterations,
        "elapsed_seconds": time.monotonic() - started,
        "baseline_v53_auc": float(roc_auc_score(vals, baseline)),
        "baseline_v53_weighted_auc": _weighted_auc(vals, baseline, ref["events"]),
        "best_trial": best,
        "trials": trials,
        "note": "No fresh holdout is available; this run is only for diagnosing sparse/stale-history weaknesses.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a development-only stronger ALS exploration")
    parser.add_argument("--days", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--quick", action="store_true", help="Reduce to a single dim/seed and day for smoke testing")
    args = parser.parse_args()

    if args.quick:
        args.days = [6]
        args.dims = [64]
        args.seeds = [4201]

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for day in args.days:
        for dim in args.dims:
            for seed in args.seeds:
                records.append(_fit_single(day, int(dim), int(seed)))

    best_overall = max(records, key=lambda r: max(t["weighted_auc"] for t in r["trials"]))
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, int, np.floating, float, str, bool)) or value is None:
            return value
        return str(value)

    report = {
        "experiment": "V68_stronger_user_item_als",
        "exploratory_only": True,
        "reused_selection_cohorts": True,
        "promotion_blocked": True,
        "resume_action": "Acquire an authorized non-overlapping labeled first-event holdout before any candidate promotion.",
        "best_overall": json_safe(best_overall),
        "records": [json_safe(r) for r in records],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"best": json_safe(best_overall["best_trial"]), "summary": "exploratory_only"}, indent=2))


if __name__ == "__main__":
    main()
