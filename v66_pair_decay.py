"""Exploratory time-decayed user-item propensity features.

Uses the same reused day5/day6 cohorts and frozen V53 comparator as V65.
These results are for diagnosis only and cannot establish generalization.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
from v39_regularized_als32 import weighted_auc
from v49_expert_calibration import source_paths
from v53_causal_propensity import features as v53_features
from v65_pair_propensity import BLEND_WEIGHT, _fit, _pair_features, _rank, _weights, sha256

ROOT = Path(__file__).resolve().parent
OUT = R / "v66_pair_decay_corrected"
DAY = 86400
SEED = 6601
DECAY_DAYS = (1.0, 7.0, 30.0)


def _decayed_pair_features(day, train, event_ids):
    start = int(train[:, 2].min())
    cutoff = start + day * DAY
    prior = np.asarray(train[train[:, 2] < cutoff])
    event_ids = np.asarray(event_ids)
    if not len(prior) or not len(event_ids) or len(prior) % 10:
        raise ValueError("Expected complete prior events and validation events")
    row_ix = (event_ids[:, None] * 10 + np.arange(10)[None, :]).reshape(-1)
    rows = np.asarray(train[row_ix])
    if int(prior[:, 2].max()) >= cutoff or int(rows[:, 2].min()) < cutoff:
        raise AssertionError("Decayed pair features must be strictly pre-cutoff")

    n_items = int(train[:, 1].max()) + 1
    pair_keys = (
        prior[:, 0].astype(np.uint64) * np.uint64(n_items)
        + prior[:, 1].astype(np.uint64)
    )
    keys, inverse = np.unique(pair_keys, return_inverse=True)
    target_keys = (
        rows[:, 0].astype(np.uint64) * np.uint64(n_items)
        + rows[:, 1].astype(np.uint64)
    )
    pos = np.searchsorted(keys, target_keys)
    found = pos < len(keys)
    found &= keys[np.minimum(pos, len(keys) - 1)] == target_keys

    ages = (cutoff - prior[:, 2].astype(np.float64)) / DAY
    click_labels = prior[:, 5].astype(np.float64)
    columns = []
    decay_counts = {}
    for tau in DECAY_DAYS:
        decay = np.exp(-ages / tau)
        exposure_sum = np.bincount(inverse, weights=decay, minlength=len(keys))
        click_sum = np.bincount(
            inverse, weights=decay * click_labels, minlength=len(keys)
        )
        target_scale = np.exp(-(rows[:, 2].astype(np.float64) - cutoff) / DAY / tau)
        decayed_exposures = np.zeros(len(rows), dtype=np.float64)
        decayed_clicks = np.zeros(len(rows), dtype=np.float64)
        decayed_exposures[found] = exposure_sum[pos[found]] * target_scale[found]
        decayed_clicks[found] = click_sum[pos[found]] * target_scale[found]
        columns.extend(
            [np.log1p(decayed_exposures), np.log1p(decayed_clicks)]
        )
        decay_counts[f"{tau:g}d"] = dict(
            exposure_mean=float(decayed_exposures.mean()),
            click_mean=float(decayed_clicks.mean()),
            seen_share=float(np.mean(decayed_exposures > 0)),
        )

    sample = np.linspace(0, len(rows) - 1, num=min(16, len(rows)), dtype=int)
    for ix in sample:
        matching = pair_keys == target_keys[ix]
        if not matching.any():
            expected = np.zeros(len(DECAY_DAYS) * 2)
        else:
            gap_days = (rows[ix, 2] - prior[matching, 2]).astype(np.float64) / DAY
            labels = prior[matching, 5].astype(np.float64)
            expected = np.asarray(
                [
                    value
                    for tau in DECAY_DAYS
                    for value in (
                        np.log1p(np.exp(-gap_days / tau).sum()),
                        np.log1p(np.sum(np.exp(-gap_days / tau) * labels)),
                    )
                ]
            )
        actual = np.asarray(columns, dtype=np.float64)[:, ix]
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise AssertionError(
                f"Decayed pair oracle mismatch at row {ix}: "
                f"actual={actual.tolist()} expected={expected.tolist()}"
            )

    features = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(features).all():
        raise AssertionError("Non-finite decayed pair features")
    return rows, features, dict(
        cutoff=cutoff,
        prior_rows=len(prior),
        distinct_user_item_pairs=len(keys),
        decay_days=DECAY_DAYS,
        decay_counts=decay_counts,
    )


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    if (OUT / "frozen.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {OUT}")

    required = list(
        dict.fromkeys(
            source_paths(5)
            + source_paths(6)
            + [
                R / "train.npy",
                R / "train_hist.npy",
                R / "train_times.npy",
                R / "evaluation_hist.npy",
                R / "v48_history_calibrator.pkl",
                R / "v53_causal_propensity_day5.npz",
                R / "v53_causal_propensity_day6.npz",
                R / "v65_pair_propensity" / "day5_oof.npz",
                ROOT / "v65_pair_propensity.py",
                ROOT / "v66_pair_decay.py",
            ]
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen experiment inputs: {missing}")

    frozen = dict(
        experiment="V66_time_decayed_user_item_propensity",
        status="exploratory_reused_cohorts_not_promotion_evidence",
        hypothesis="Recent user-item exposures and clicks carry more signal than lifetime pair totals, particularly when preferences drift.",
        fit="day5 first-event rows; two complementary user folds for OOF",
        test="day6 first-event-per-user cohort; reused development date",
        comparator="final saved V53 day5/day6 predictions",
        feature_cutoff="strictly before each day cutoff; direct raw-row decay oracle checked on sampled target pairs",
        added_features="log decayed pair exposures and clicks at 1, 7, and 30 day exponential time constants",
        blend_weight=BLEND_WEIGHT,
        model_params=dict(iterations=400, depth=6, learning_rate=0.04, seed=SEED),
        input_sha256={path.name: sha256(path) for path in required},
        runtime_device="local CPU; no GPU",
    )
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2))

    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        spline = pickle.load(stream)["spline"]
    train = np.load(R / "train.npy", mmap_mode="r")
    hist = np.load(R / "train_hist.npy", mmap_mode="r")
    times = np.load(R / "train_times.npy", mmap_mode="r")
    eval_hist = np.load(R / "evaluation_hist.npy", mmap_mode="r")
    z5, x5, users5, groups5, _ = v53_features(5, train, hist, times, spline)
    z6, x6, _, groups6, _ = v53_features(6, train, hist, times, spline)
    baseline5 = np.load(R / "v53_causal_propensity_day5.npz")
    baseline6 = np.load(R / "v53_causal_propensity_day6.npz")
    if not (
        np.array_equal(z5["events"], baseline5["events"])
        and np.array_equal(z5["y"], baseline5["y"])
        and np.array_equal(z6["events"], baseline6["events"])
        and np.array_equal(z6["y"], baseline6["y"])
    ):
        raise AssertionError("Feature labels/events differ from saved final V53 cohorts")

    rows5, static5, meta_static5 = _pair_features(5, train, z5["events"][::10])
    rows6, static6, meta_static6 = _pair_features(6, train, z6["events"][::10])
    decay_rows5, decay5, meta_decay5 = _decayed_pair_features(
        5, train, z5["events"][::10]
    )
    decay_rows6, decay6, meta_decay6 = _decayed_pair_features(
        6, train, z6["events"][::10]
    )
    if not np.array_equal(rows5, decay_rows5) or not np.array_equal(
        rows6, decay_rows6
    ):
        raise AssertionError("Static and decayed pair rows are misaligned")
    x5 = np.column_stack((x5, static5, decay5)).astype(np.float32)
    x6 = np.column_stack((x6, static6, decay6)).astype(np.float32)
    y5 = np.asarray(z5["y"], dtype=np.int8)
    y6 = np.asarray(z6["y"], dtype=np.int8)
    w5 = _weights(groups5, eval_hist)
    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        fit_users = np.asarray(pickle.load(stream)["fit_users"])

    oof = np.full(len(y5), np.nan, dtype=np.float32)
    for fold, mask in enumerate(
        [np.isin(users5, fit_users), ~np.isin(users5, fit_users)]
    ):
        if not mask.any() or mask.all():
            raise AssertionError("Day5 user-crossfit fold is empty or degenerate")
        model = _fit(x5[mask], y5[mask], w5[mask], SEED + fold)
        held = ~mask
        oof[held] = model.predict_proba(x5[held])[:, 1].astype(np.float32)
        model.save_model(str(OUT / f"day5_fold{fold}.cbm"))
        print(
            json.dumps(
                dict(
                    stage="day5_user_oof",
                    fold=fold,
                    fit_rows=int(mask.sum()),
                    held_rows=int(held.sum()),
                    elapsed_seconds=round(time.monotonic() - started, 1),
                )
            ),
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise AssertionError("OOF prediction coverage is incomplete")

    pred5 = _rank(0.80 * _rank(baseline5["pred"]) + BLEND_WEIGHT * _rank(oof))
    report5 = dict(
        day=5,
        protocol="full first-event day5 user crossfit; exploratory/reused",
        ordinary_v53=float(roc_auc_score(y5, baseline5["pred"])),
        ordinary_candidate=float(roc_auc_score(y5, pred5)),
        adjusted_v53=weighted_auc(baseline5, baseline5["pred"]),
        adjusted_candidate=weighted_auc(baseline5, pred5),
    )
    report5["gain_adjusted"] = report5["adjusted_candidate"] - report5["adjusted_v53"]
    np.savez_compressed(
        OUT / "day5_oof.npz",
        events=baseline5["events"],
        y=y5,
        v53=baseline5["pred"],
        raw=oof,
        pred=pred5,
    )

    full = _fit(x5, y5, w5, SEED)
    full.save_model(str(OUT / "day5_full.cbm"))
    raw6 = full.predict_proba(x6)[:, 1].astype(np.float32)
    pred6 = _rank(0.80 * _rank(baseline6["pred"]) + BLEND_WEIGHT * _rank(raw6))
    report6 = dict(
        day=6,
        protocol="train day5 only, then score future day6 first-event cohort; exploratory/reused",
        ordinary_v53=float(roc_auc_score(y6, baseline6["pred"])),
        ordinary_candidate=float(roc_auc_score(y6, pred6)),
        adjusted_v53=weighted_auc(baseline6, baseline6["pred"]),
        adjusted_candidate=weighted_auc(baseline6, pred6),
    )
    report6["gain_adjusted"] = report6["adjusted_candidate"] - report6["adjusted_v53"]
    np.savez_compressed(
        OUT / "day6_future.npz",
        events=baseline6["events"],
        y=y6,
        v53=baseline6["pred"],
        raw=raw6,
        pred=pred6,
    )

    report = dict(
        frozen=frozen,
        pair_feature_generation=dict(
            day5_static=meta_static5,
            day6_static=meta_static6,
            day5_decay=meta_decay5,
            day6_decay=meta_decay6,
        ),
        day5=report5,
        day6=report6,
        accepted_as_best=False,
        promotion_status="EXPLORATORY_ONLY: reused cohorts; fresh labeled holdout unavailable",
        target_0_85_met=False,
        runtime_seconds=round(time.monotonic() - started, 3),
        device="CPU",
        official_auc=None,
        artifacts=str(OUT),
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
