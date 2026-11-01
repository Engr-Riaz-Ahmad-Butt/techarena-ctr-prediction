"""Exploratory event-grouped pairwise ranker over causal V53 features."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
from v39_regularized_als32 import weighted_auc
from v49_expert_calibration import source_paths
from v53_causal_propensity import features as v53_features
from v65_pair_propensity import _rank, _weights, sha256

ROOT = Path(__file__).resolve().parent
OUT = R / "v67_event_pairwise_ranker_corrected"
SEED = 6701
BLEND_WEIGHT = 0.20
PARAMS = dict(
    iterations=250,
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=12.0,
    loss_function="PairLogitPairwise",
    random_seed=SEED,
    thread_count=4,
    verbose=False,
    allow_writing_files=False,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    random_strength=0.5,
)


def _ordered(indices, event_ids):
    return indices[np.argsort(event_ids[indices], kind="stable")]


def _pool(x, y, event_ids, weights, indices):
    indices = _ordered(indices, event_ids)
    return Pool(
        x[indices],
        label=y[indices],
        group_id=event_ids[indices],
        group_weight=weights[indices],
    ), indices


def _fit_ranker(x, y, event_ids, weights, indices, seed):
    model = CatBoostRanker(**dict(PARAMS, random_seed=seed))
    pool, _ = _pool(x, y, event_ids, weights, indices)
    model.fit(pool)
    return model


def _check_events(z, groups):
    if len(z["events"]) != len(z["y"]) or len(z["events"]) % 10:
        raise AssertionError("Expected complete ten-candidate event groups")
    event_ids = np.asarray(z["events"])
    labels = np.asarray(z["y"])
    if not np.all(event_ids.reshape(-1, 10) == event_ids.reshape(-1, 10)[:, :1]):
        raise AssertionError("Candidate rows are not contiguous by event")
    positives = labels.reshape(-1, 10).sum(axis=1)
    if np.any(positives > 1):
        raise AssertionError("An event contains more than one clicked item")
    if not np.all(groups.reshape(-1, 10) == groups.reshape(-1, 10)[:, :1]):
        raise AssertionError("Event candidate rows have inconsistent history weights")
    return event_ids, labels, int(np.count_nonzero(positives == 0))


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
                ROOT / "v67_event_pairwise_ranker.py",
            ]
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen experiment inputs: {missing}")
    frozen = dict(
        experiment="V67_event_grouped_pairwise_ranker",
        status="exploratory_reused_cohorts_not_promotion_evidence",
        hypothesis="Learning pairwise preferences within each ten-item event may better match the one-click choice task than pointwise binary log-loss.",
        fit="day5 first-event rows; two complementary user folds for OOF",
        test="day6 first-event-per-user cohort; reused development date",
        comparator="final saved V53 day5/day6 predictions",
        loss="CatBoost PairLogitPairwise; event ID is the group",
        parameters=PARAMS,
        blend_weight=BLEND_WEIGHT,
        feature_cutoff="unchanged causal V53 feature builder",
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
    event5, y5, zero_click_events5 = _check_events(z5, groups5)
    event6, y6, zero_click_events6 = _check_events(z6, groups6)
    if not (
        np.array_equal(event5, baseline5["events"])
        and np.array_equal(y5, baseline5["y"])
        and np.array_equal(event6, baseline6["events"])
        and np.array_equal(y6, baseline6["y"])
    ):
        raise AssertionError("Feature labels/events differ from saved final V53 cohorts")

    weights5 = _weights(groups5, eval_hist)
    weights6 = _weights(groups6, eval_hist)
    if not (
        np.allclose(weights5.reshape(-1, 10), weights5.reshape(-1, 10)[:, :1])
        and np.allclose(weights6.reshape(-1, 10), weights6.reshape(-1, 10)[:, :1])
    ):
        raise AssertionError("Pairwise group weights must be constant within each event")

    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        fit_users = np.asarray(pickle.load(stream)["fit_users"])
    fold_masks = [np.isin(users5, fit_users), ~np.isin(users5, fit_users)]
    oof = np.full(len(y5), np.nan, dtype=np.float32)
    for fold, mask in enumerate(fold_masks):
        if not mask.any() or mask.all():
            raise AssertionError("Day5 user-crossfit fold is empty or degenerate")
        fit_indices = np.flatnonzero(mask)
        held_indices = _ordered(np.flatnonzero(~mask), event5)
        model = _fit_ranker(
            x5, y5, event5, weights5, fit_indices, SEED + fold
        )
        held_pool, held_indices = _pool(
            x5, y5, event5, weights5, held_indices
        )
        oof[held_indices] = model.predict(held_pool).astype(np.float32)
        model.save_model(str(OUT / f"day5_fold{fold}.cbm"))
        print(
            json.dumps(
                dict(
                    stage="day5_user_oof",
                    fold=fold,
                    fit_rows=int(mask.sum()),
                    held_rows=int((~mask).sum()),
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

    full = _fit_ranker(
        x5, y5, event5, weights5, np.arange(len(y5)), SEED
    )
    full.save_model(str(OUT / "day5_full.cbm"))
    eval_pool, order6 = _pool(
        x6, y6, event6, weights6, np.arange(len(y6))
    )
    raw6 = np.empty(len(y6), dtype=np.float32)
    raw6[order6] = full.predict(eval_pool).astype(np.float32)
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
        zero_click_events=dict(day5=zero_click_events5, day6=zero_click_events6),
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
