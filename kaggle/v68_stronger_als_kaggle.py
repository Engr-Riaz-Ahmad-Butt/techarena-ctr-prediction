"""Kaggle-ready version of the strong ALS exploratory sweep.

This is intentionally development-only and reuses the same validation cohorts as the
local V65-V67 experiments. It never creates a competition submission or claims a
fresh-holdout result.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def discover_repo_root() -> Path:
    candidates = [
        Path.cwd(),
        Path("/kaggle/working"),
        Path("/kaggle/working/techarena-ctr-prediction"),
        Path("/content"),
        Path("/workspace"),
    ]
    for candidate in candidates:
        if (candidate / "config.py").exists() and (candidate / "v68_stronger_als.py").exists():
            return candidate
    # Fall back to the current working directory if the repo is mounted there.
    for candidate in candidates:
        if (candidate / "config.py").exists():
            return candidate
    raise FileNotFoundError("Could not find the repository root. Mount the repo in the notebook working directory.")


REPO = discover_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ARTIFACT_DIR as DEFAULT_ARTIFACT_DIR
from imp15_als import prepare, split, cutoff_of, score, augmented, solve_rows, nu, ni

ROOT = REPO
OUT = Path(os.environ.get("CTR_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_DIMS = [64, 128, 256]
DEFAULT_SEEDS = [101, 202, 303, 404, 505]
BETA_GRID = [0.05, 0.10, 0.20]


def _rank(values):
    return rankdata(values, method="average") / (len(values) + 1.0)


def _weighted_auc(y, pred, events):
    hist = np.load(OUT.parent / "train_hist.npy", mmap_mode="r")
    eval_hist = np.load(OUT.parent / "evaluation_hist.npy", mmap_mode="r")
    eval_groups = np.digitize((eval_hist >= 0).sum(axis=1), [0, 4, 9, 19], right=True)
    val_groups = np.digitize((hist[events] >= 0).sum(axis=1), [0, 4, 9, 19], right=True)
    target = np.bincount(eval_groups, minlength=5).astype(np.float64)
    target /= target.sum()
    source = np.bincount(val_groups, minlength=5).astype(np.float64)
    source /= source.sum()
    weights = np.divide(target, source, out=np.zeros(5), where=source > 0)[val_groups]
    weights /= weights.mean()
    return float(roc_auc_score(y, pred, sample_weight=weights))


def _initial_v(dim, day, seed):
    base = np.load(OUT.parent / f"imp15_model_day{day}.npz")
    v = base["V"].astype(np.float64)
    if v.shape[1] < dim:
        rng = np.random.default_rng(seed)
        extra = rng.normal(0.0, 0.05, size=(v.shape[0], dim - v.shape[1]))
        v = np.concatenate([v, extra], axis=1)
    else:
        v = v[:, :dim].copy()
    return v


def fit_single(day, dim, seed, iterations=6):
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
    ref = np.load(OUT.parent / f"v53_causal_propensity_day{day}.npz")
    if not (np.array_equal(val[:, 6], ref["events"]) and np.array_equal(val[:, 5], ref["y"])):
        raise AssertionError(f"Validation rows do not match the saved V53 cohort for day {day}")

    y = ref["y"]
    baseline = ref["pred"]
    baseline_rank = _rank(baseline)
    raw = score(val, u, v, bu, bi, g, exposures)
    raw_rank = _rank(raw)

    best = None
    trials = []
    for weight in BETA_GRID:
        pred = (1.0 - weight) * baseline_rank + weight * raw_rank
        pred = np.asarray(pred, dtype=np.float64)
        auc = float(roc_auc_score(y, pred))
        weighted_auc = _weighted_auc(y, pred, ref["events"])
        gain = weighted_auc - _weighted_auc(y, baseline, ref["events"])
        entry = {
            "day": day,
            "dim": dim,
            "seed": seed,
            "weight": weight,
            "auc": auc,
            "weighted_auc": weighted_auc,
            "gain_vs_v53": gain,
        }
        trials.append(entry)
        if best is None or weighted_auc > best["weighted_auc"]:
            best = entry.copy()

    return {
        "experiment": "V68_stronger_user_item_als",
        "exploratory_only": True,
        "promotable": False,
        "day": day,
        "dim": dim,
        "seed": seed,
        "iterations": iterations,
        "elapsed_seconds": time.monotonic() - started,
        "baseline_v53_weighted_auc": _weighted_auc(y, baseline, ref["events"]),
        "best_trial": best,
        "trials": trials,
        "note": "Development-only evidence on reused first-event cohorts; no fresh holdout exists.",
    }


def main():
    days = [int(x) for x in os.environ.get("KAGGLE_V68_DAYS", "5,6").split(",") if x.strip()]
    dims = [int(x) for x in os.environ.get("KAGGLE_V68_DIMS", "64,128,256").split(",") if x.strip()]
    seeds = [int(x) for x in os.environ.get("KAGGLE_V68_SEEDS", "101,202,303,404,505").split(",") if x.strip()]

    records = []
    for day in days:
        for dim in dims:
            for seed in seeds:
                records.append(fit_single(day, dim, seed))

    best = max(records, key=lambda r: max(t["weighted_auc"] for t in r["trials"]))
    report = {
        "experiment": "V68_stronger_user_item_als",
        "exploratory_only": True,
        "resume_action": "Acquire an authorized non-overlapping labeled future holdout before candidate selection.",
        "promotion_blocked": True,
        "best": best,
        "records": records,
    }
    output_path = OUT / "v68_stronger_als_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"best": best["best_trial"], "status": "exploratory_only"}, indent=2))


if __name__ == "__main__":
    main()
