"""Exploratory user-item historical exposure propensity correction.

The model is trained only on first-event day5 labels and tested on the later
first-event day6 cohort. Both cohorts have been repeatedly reused, so this is
development evidence only and cannot promote a model.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
from v39_regularized_als32 import weighted_auc
from v49_expert_calibration import source_paths
from v53_causal_propensity import features as v53_features

ROOT = Path(__file__).resolve().parent
OUT = R / "v65_pair_propensity"
DAY = 86400
SEED = 6501
BLEND_WEIGHT = 0.20
PARAMS = dict(
    iterations=400,
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=12.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    thread_count=4,
    verbose=False,
    allow_writing_files=False,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    random_strength=0.5,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(values):
    return rankdata(values, method="average") / (len(values) + 1.0)


def _pair_features(day, train, event_ids):
    start = int(train[:, 2].min())
    cutoff = start + day * DAY
    prior = np.asarray(train[train[:, 2] < cutoff])
    event_ids = np.asarray(event_ids)
    if not len(prior) or not len(event_ids):
        raise ValueError("Expected prior rows and complete 10-row validation events")
    row_ix = (event_ids[:, None] * 10 + np.arange(10)[None, :]).reshape(-1)
    rows = np.asarray(train[row_ix])
    if int(prior[:, 2].max()) >= cutoff or int(rows[:, 2].min()) < cutoff:
        raise AssertionError("Pair features must use only rows before the validation cutoff")
    if len(prior) % 10:
        raise AssertionError("Prior data must contain complete 10-candidate events")

    n_items = int(train[:, 1].max()) + 1
    prior_users = prior[:, 0].astype(np.uint64)
    prior_items = prior[:, 1].astype(np.uint64)
    keys = prior_users * np.uint64(n_items) + prior_items
    unique_keys, inverse, exposures = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    clicks = np.bincount(
        inverse,
        weights=prior[:, 5].astype(np.float64),
        minlength=len(unique_keys),
    )
    last_seen = np.full(len(unique_keys), -1, dtype=np.int64)
    np.maximum.at(last_seen, inverse, prior[:, 2].astype(np.int64))

    target_keys = (
        rows[:, 0].astype(np.uint64) * np.uint64(n_items)
        + rows[:, 1].astype(np.uint64)
    )
    pos = np.searchsorted(unique_keys, target_keys)
    found = pos < len(unique_keys)
    found &= unique_keys[np.minimum(pos, len(unique_keys) - 1)] == target_keys
    pair_n = np.zeros(len(rows), dtype=np.float64)
    pair_c = np.zeros(len(rows), dtype=np.float64)
    pair_last = np.full(len(rows), -1, dtype=np.int64)
    pair_n[found] = exposures[pos[found]]
    pair_c[found] = clicks[pos[found]]
    pair_last[found] = last_seen[pos[found]]

    event_clicks = prior[:, 5].reshape(-1, 10).sum(axis=1)
    user_events = prior[::10, 0].astype(np.int64)
    n_users = int(train[:, 0].max()) + 1
    user_n = np.bincount(user_events, minlength=n_users).astype(np.float64)
    user_c = np.bincount(
        user_events, weights=event_clicks, minlength=n_users
    )
    global_event_rate = float(event_clicks.mean())
    user_rate = (
        (user_c + 20.0 * global_event_rate) / (user_n + 20.0) / 10.0
    )
    prior_item_rate = float(prior[:, 5].mean())
    smooth_rate = (pair_c + 10.0 * user_rate[rows[:, 0]]) / (pair_n + 10.0)
    pair_age = np.where(
        pair_last >= 0,
        np.maximum(rows[:, 2].astype(np.int64) - pair_last, 0) / DAY,
        30.0,
    )

    extra = np.column_stack(
        [
            np.log1p(pair_n),
            np.log1p(pair_c),
            logit(np.clip(smooth_rate, 1e-5, 1.0 - 1e-5)),
            np.log1p(pair_age),
            (pair_n > 0).astype(np.float32),
            (pair_c > 0).astype(np.float32),
            np.log1p(user_rate[rows[:, 0]]),
        ]
    ).astype(np.float32)
    if not np.isfinite(extra).all():
        raise AssertionError("Non-finite pair propensity features")

    # A direct vectorized oracle for sampled target pairs guards key packing.
    sample = np.linspace(0, len(rows) - 1, num=min(128, len(rows)), dtype=int)
    if sample.size:
        sample_keys = np.unique(target_keys[sample])
        selected = np.isin(keys, sample_keys)
        oracle_keys, oracle_inverse, oracle_exposures = np.unique(
            keys[selected], return_inverse=True, return_counts=True
        )
        oracle_clicks = np.bincount(
            oracle_inverse,
            weights=prior[selected, 5].astype(np.float64),
            minlength=len(oracle_keys),
        )
        oracle_last = np.full(len(oracle_keys), -1, dtype=np.int64)
        np.maximum.at(
            oracle_last, oracle_inverse, prior[selected, 2].astype(np.int64)
        )
        for ix in sample:
            oracle_pos = np.searchsorted(oracle_keys, target_keys[ix])
            if (
                oracle_pos < len(oracle_keys)
                and oracle_keys[oracle_pos] == target_keys[ix]
            ):
                value = (
                    int(oracle_exposures[oracle_pos]),
                    float(oracle_clicks[oracle_pos]),
                    int(oracle_last[oracle_pos]),
                )
            else:
                value = (0, 0.0, -1)
            if (
                pair_n[ix] != value[0]
                or pair_c[ix] != value[1]
                or pair_last[ix] != value[2]
            ):
                raise AssertionError("Pair feature lookup disagrees with raw-row oracle")
    return rows, extra, dict(
        cutoff=cutoff,
        prior_rows=len(prior),
        distinct_user_item_pairs=len(unique_keys),
        pair_seen_share=float(np.mean(pair_n > 0)),
        pair_click_seen_share=float(np.mean(pair_c > 0)),
        prior_global_item_rate=prior_item_rate,
    )


def _weights(groups, evaluation_hist):
    eval_groups = np.digitize(
        (evaluation_hist >= 0).sum(axis=1), [0, 4, 9, 19], right=True
    )
    target = np.bincount(eval_groups, minlength=5).astype(np.float64)
    target /= target.sum()
    source = np.bincount(groups[::10], minlength=5).astype(np.float64)
    source /= source.sum()
    w = np.divide(target, source, out=np.zeros(5), where=source > 0)[groups]
    w /= w.mean()
    return w.astype(np.float32)


def _fit(x, y, weights, seed):
    params = dict(PARAMS, random_seed=seed)
    model = CatBoostClassifier(**params)
    model.fit(x, y, sample_weight=weights)
    return model


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
                ROOT / "v65_pair_propensity.py",
            ]
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen experiment inputs: {missing}")
    frozen = dict(
        experiment="V65_user_item_prior_exposure_propensity",
        status="exploratory_reused_cohorts_not_promotion_evidence",
        hypothesis="User-item pair exposure/click history, smoothed with causal user event propensity, adds sparse affinity information absent from aggregate user/item rates.",
        fit="day5 first-event rows; two complementary user folds for OOF",
        test="day6 first-event-per-user cohort; the same reused local development date",
        comparator="final saved V53 day5/day6 prediction arrays",
        feature_cutoff="strictly before each day cutoff; raw-row oracle checked on sampled target pairs",
        blend_weight=BLEND_WEIGHT,
        model_params=PARAMS,
        seeds=[SEED],
        metric="evaluation-history-adjusted ROC AUC, plus ordinary AUC",
        promotion="impossible on these reused cohorts; require new labeled holdout and at least +0.0005 on both selection cohorts",
        input_sha256={path.name: sha256(path) for path in required},
        library="CatBoost",
    )
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=2))

    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        spline = pickle.load(stream)["spline"]
    train = np.load(R / "train.npy", mmap_mode="r")
    hist = np.load(R / "train_hist.npy", mmap_mode="r")
    times = np.load(R / "train_times.npy", mmap_mode="r")
    eval_hist = np.load(R / "evaluation_hist.npy", mmap_mode="r")
    z5, x5, users5, groups5, _ = v53_features(5, train, hist, times, spline)
    z6, x6, users6, groups6, _ = v53_features(6, train, hist, times, spline)

    baseline5 = np.load(R / "v53_causal_propensity_day5.npz")
    baseline6 = np.load(R / "v53_causal_propensity_day6.npz")
    if not (
        np.array_equal(z5["events"], baseline5["events"])
        and np.array_equal(z5["y"], baseline5["y"])
        and np.array_equal(z6["events"], baseline6["events"])
        and np.array_equal(z6["y"], baseline6["y"])
    ):
        raise AssertionError("Feature labels/events differ from saved final V53 cohorts")
    rows5, pair5, meta5 = _pair_features(5, train, z5["events"][::10])
    rows6, pair6, meta6 = _pair_features(6, train, z6["events"][::10])
    if int(rows5[:, 2].max()) >= int(rows6[:, 2].min()):
        raise AssertionError("Day5 training must precede day6 evaluation timestamps")

    x5 = np.column_stack((x5, pair5)).astype(np.float32)
    x6 = np.column_stack((x6, pair6)).astype(np.float32)
    y5 = np.asarray(z5["y"], dtype=np.int8)
    y6 = np.asarray(z6["y"], dtype=np.int8)
    w5 = _weights(groups5, eval_hist)
    w6 = _weights(groups6, eval_hist)

    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        fit_users = np.asarray(pickle.load(stream)["fit_users"])
    fold_masks = [np.isin(users5, fit_users), ~np.isin(users5, fit_users)]
    oof = np.full(len(y5), np.nan, dtype=np.float32)
    for fold, mask in enumerate(fold_masks):
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

    pred5 = _rank(
        0.80 * _rank(baseline5["pred"]) + BLEND_WEIGHT * _rank(oof)
    )
    report5 = dict(
        day=5,
        protocol="full first-event day5 user crossfit; exploratory/reused",
        ordinary_v53=float(roc_auc_score(y5, baseline5["pred"])),
        ordinary_candidate=float(roc_auc_score(y5, pred5)),
        adjusted_v53=float(weighted_auc(baseline5, baseline5["pred"])),
        adjusted_candidate=float(weighted_auc(baseline5, pred5)),
        gain_adjusted=float(
            weighted_auc(baseline5, pred5) - weighted_auc(baseline5, baseline5["pred"])
        ),
    )
    np.savez_compressed(
        OUT / "day5_oof.npz",
        events=baseline5["events"],
        y=y5,
        v53=baseline5["pred"],
        raw=oof,
        pred=pred5,
    )
    print(json.dumps(report5), flush=True)

    full = _fit(x5, y5, w5, SEED)
    full.save_model(str(OUT / "day5_full.cbm"))
    raw6 = full.predict_proba(x6)[:, 1].astype(np.float32)
    pred6 = _rank(
        0.80 * _rank(baseline6["pred"]) + BLEND_WEIGHT * _rank(raw6)
    )
    report6 = dict(
        day=6,
        protocol="train day5 only, then score future day6 first-event cohort; exploratory/reused",
        ordinary_v53=float(roc_auc_score(y6, baseline6["pred"])),
        ordinary_candidate=float(roc_auc_score(y6, pred6)),
        ordinary_gain=float(
            roc_auc_score(y6, pred6) - roc_auc_score(y6, baseline6["pred"])
        ),
        adjusted_v53=float(weighted_auc(baseline6, baseline6["pred"])),
        adjusted_candidate=float(weighted_auc(baseline6, pred6)),
        gain_adjusted=float(
            weighted_auc(baseline6, pred6)
            - weighted_auc(baseline6, baseline6["pred"])
        ),
    )
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
        pair_feature_generation=dict(day5=meta5, day6=meta6),
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
