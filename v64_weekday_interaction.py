"""Frozen day5-to-day6 nonlinear correction over V53 causal features.

The only supervised fit is on day5 first-future-event labels. Day6 labels are
used only after fitting to score the frozen future-event cohort. Temporal
features are derived from pre-cutoff histories and UTC timestamps.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R
import v53_causal_propensity as v53

ROOT = Path(__file__).resolve().parent
DAY = 86400
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BLEND_WEIGHT = 0.25
MODEL_PARAMS = dict(
    iterations=600,
    depth=7,
    learning_rate=0.04,
    l2_leaf_reg=12.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=6401,
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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_weekday(timestamp):
    """Return Monday=0..Sunday=6 for Unix timestamps in UTC."""
    return ((np.asarray(timestamp, dtype=np.int64) // DAY + 3) % 7).astype(np.int8)


def _logit(rate):
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    return np.log(rate / (1.0 - rate)).astype(np.float32)


def _one_hot(values, width):
    values = np.asarray(values, dtype=np.int64)
    if np.any((values < 0) | (values >= width)):
        raise ValueError(f"Category outside [0, {width}): {values.min()}..{values.max()}")
    return np.eye(width, dtype=np.float32)[values]


def _features(day, t, h, ht, spline):
    z, base_x, users, groups, names = v53.features(day, t, h, ht, spline)
    events = np.asarray(z["events"])
    if len(events) % 10:
        raise AssertionError("Frozen benchmark rows must be complete 10-candidate events")
    row_indices = events * 10 + np.tile(np.arange(10), len(events) // 10)
    rows = np.asarray(t[row_indices])
    if not np.array_equal(rows[:, 6], events):
        raise AssertionError("V53 event-to-candidate row alignment changed")
    if len(rows) != len(z["y"]):
        raise AssertionError("V53 feature/label row count mismatch")

    start = int(t[:, 2].min())
    cutoff = start + day * DAY
    prior = np.asarray(t[t[:, 2] < cutoff])
    if len(prior) == 0 or int(prior[:, 2].max()) >= cutoff or int(rows[:, 2].min()) < cutoff:
        raise AssertionError("Temporal feature cutoff is not strictly causal")
    if len(prior) % 10:
        raise AssertionError("Prior history contains a partial 10-candidate event")

    history_events = prior[::10]
    event_clicks = prior[:, 5].reshape(-1, 10).sum(axis=1)
    prior_dow = utc_weekday(history_events[:, 2])
    target_dow = utc_weekday(rows[:, 2])
    user_ids = rows[:, 0].astype(np.int64)
    item_ids = rows[:, 1].astype(np.int64)
    n_users = int(max(t[:, 0].max(), user_ids.max())) + 1
    n_items = int(max(t[:, 1].max(), item_ids.max())) + 1
    global_event_rate = float(event_clicks.mean())
    global_item_rate = float(prior[:, 5].mean())

    user_n = np.bincount(history_events[:, 0], minlength=n_users).astype(np.float64)
    user_c = np.bincount(history_events[:, 0], weights=event_clicks, minlength=n_users)
    user_all = (user_c + 20.0 * global_event_rate) / (user_n + 20.0)
    user_key = history_events[:, 0].astype(np.int64) * 7 + prior_dow
    user_dow_n = np.bincount(user_key, minlength=n_users * 7).astype(np.float64)
    user_dow_c = np.bincount(user_key, weights=event_clicks, minlength=n_users * 7)
    target_user_key = user_ids * 7 + target_dow
    user_dow_rate = (user_dow_c[target_user_key] + 8.0 * user_all[user_ids]) / (user_dow_n[target_user_key] + 8.0)

    prior_item_dow = utc_weekday(prior[:, 2])
    item_n = np.bincount(prior[:, 1], minlength=n_items).astype(np.float64)
    item_c = np.bincount(prior[:, 1], weights=prior[:, 5], minlength=n_items)
    item_all = (item_c + 100.0 * global_item_rate) / (item_n + 100.0)
    item_key = prior[:, 1].astype(np.int64) * 7 + prior_item_dow
    item_dow_n = np.bincount(item_key, minlength=n_items * 7).astype(np.float64)
    item_dow_c = np.bincount(item_key, weights=prior[:, 5], minlength=n_items * 7)
    target_item_key = item_ids * 7 + target_dow
    item_dow_rate = (item_dow_c[target_item_key] + 100.0 * item_all[item_ids]) / (item_dow_n[target_item_key] + 100.0)

    # Pool weekday/weekend rates as a lower-variance temporal fallback.
    prior_weekend = prior_dow >= 5
    prior_item_weekend = prior_item_dow >= 5
    user_weekend_n = np.bincount(history_events[prior_weekend, 0], minlength=n_users).astype(np.float64)
    user_weekend_c = np.bincount(history_events[prior_weekend, 0], weights=event_clicks[prior_weekend], minlength=n_users)
    user_weekend_rate = (user_weekend_c[user_ids] + 8.0 * user_all[user_ids]) / (user_weekend_n[user_ids] + 8.0)
    item_weekend_n = np.bincount(prior[prior_item_weekend, 1], minlength=n_items).astype(np.float64)
    item_weekend_c = np.bincount(prior[prior_item_weekend, 1], weights=prior[prior_item_weekend, 5], minlength=n_items)
    item_weekend_rate = (item_weekend_c[item_ids] + 100.0 * item_all[item_ids]) / (item_weekend_n[item_ids] + 100.0)

    target_hour = ((rows[:, 2] % DAY) // 3600).astype(np.int8)
    hour_bin = np.minimum(target_hour // 4, 5)
    dow_hot = _one_hot(target_dow, 7)
    hour_hot = _one_hot(hour_bin, 6)
    bucket = rows[:, 3].astype(np.int64)
    bucket_hot = _one_hot(bucket, 4)
    dow_hour = (dow_hot[:, :, None] * hour_hot[:, None, :]).reshape(len(rows), 42)
    dow_bucket = (dow_hot[:, :, None] * bucket_hot[:, None, :]).reshape(len(rows), 28)
    hour_bucket = (hour_hot[:, :, None] * bucket_hot[:, None, :]).reshape(len(rows), 24)
    weekend = (target_dow >= 5).astype(np.float32)

    baseline_rank = rankdata(np.asarray(z["pred"]), method="average").astype(np.float32) / (len(rows) + 1.0)
    user_dow_logit = _logit(user_dow_rate)
    item_dow_logit = _logit(item_dow_rate)
    user_weekend_logit = _logit(user_weekend_rate)
    item_weekend_logit = _logit(item_weekend_rate)

    extra = [
        dow_hot,
        weekend[:, None],
        hour_hot,
        np.column_stack((np.sin(2 * np.pi * target_hour / 24), np.cos(2 * np.pi * target_hour / 24))).astype(np.float32),
        dow_hour,
        bucket_hot,
        dow_bucket,
        hour_bucket,
        user_dow_logit[:, None],
        item_dow_logit[:, None],
        np.log1p(user_dow_n[target_user_key]).astype(np.float32)[:, None],
        np.log1p(item_dow_n[target_item_key]).astype(np.float32)[:, None],
        user_weekend_logit[:, None],
        item_weekend_logit[:, None],
        np.log1p(user_weekend_n[user_ids]).astype(np.float32)[:, None],
        np.log1p(item_weekend_n[item_ids]).astype(np.float32)[:, None],
        (user_dow_logit * item_dow_logit)[:, None],
        (user_dow_logit - item_dow_logit)[:, None],
        (user_dow_logit * baseline_rank)[:, None],
        (item_dow_logit * baseline_rank)[:, None],
        (user_weekend_logit * baseline_rank)[:, None],
        (item_weekend_logit * baseline_rank)[:, None],
        (weekend * baseline_rank)[:, None],
        baseline_rank[:, None] * dow_hot,
        baseline_rank[:, None] * hour_hot,
        user_dow_logit[:, None] * hour_hot,
        item_dow_logit[:, None] * hour_hot,
        user_all[user_ids].astype(np.float32)[:, None] * bucket_hot,
        item_all[item_ids].astype(np.float32)[:, None] * bucket_hot,
        user_dow_logit[:, None] * bucket_hot,
        item_dow_logit[:, None] * bucket_hot,
    ]
    extra = np.column_stack(extra).astype(np.float32, copy=False)
    x = np.column_stack((np.asarray(base_x, dtype=np.float32), extra))
    if not np.isfinite(x).all():
        raise AssertionError("Non-finite nonlinear-correction features")
    if np.any(groups.reshape(-1, 10) != groups.reshape(-1, 10)[:, :1]):
        raise AssertionError("History group labels differ within an event")
    metadata = dict(
        rows=rows,
        cutoff=cutoff,
        weekday_counts=np.bincount(target_dow[::10], minlength=7).tolist(),
        feature_names=list(names) + [f"temporal_interaction_{i}" for i in range(extra.shape[1])],
    )
    return z, x, users, groups, metadata


def _distribution_weights(groups, evaluation_hist):
    eval_groups = np.digitize((evaluation_hist >= 0).sum(axis=1), [0, 4, 9, 19], right=True)
    target = np.bincount(eval_groups, minlength=5).astype(np.float64)
    target /= target.sum()
    source = np.bincount(groups[::10], minlength=5).astype(np.float64)
    source /= source.sum()
    weights_by_group = np.divide(target, source, out=np.zeros(5), where=source > 0)
    weights = weights_by_group[groups]
    weights /= weights.mean()
    return weights.astype(np.float32), target.tolist(), source.tolist()


def _weighted_auc(y, pred, weights):
    return float(roc_auc_score(y, pred, sample_weight=weights))


def _check_first_future_event(day, z, meta, train):
    rows = meta["rows"]
    event_rows = rows[::10]
    users = event_rows[:, 0].astype(np.int64)
    if len(np.unique(users)) != len(users):
        raise AssertionError(f"Day{day} benchmark is not first-event-per-user")
    if int(event_rows[:, 2].min()) < meta["cutoff"]:
        raise AssertionError(f"Day{day} contains a pre-cutoff event")
    future = np.asarray(train[train[:, 2] >= meta["cutoff"]])
    future_events = future[::10]
    first_time = np.full(int(train[:, 0].max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_time, future_events[:, 0], future_events[:, 2])
    if not np.array_equal(first_time[users], event_rows[:, 2]):
        raise AssertionError(f"Day{day} cohort is not each user's first future event")
    return users


def main():
    started = time.monotonic()
    t_path = R / "train.npy"
    h_path = R / "train_hist.npy"
    ht_path = R / "train_times.npy"
    eh_path = R / "evaluation_hist.npy"
    baseline_paths = [R / f"v53_causal_propensity_day{d}.npz" for d in (5, 6)]
    spline_path = R / "v48_history_calibrator.pkl"
    source_paths = [ROOT / name for name in [
        "v53_causal_propensity.py",
        "v49_expert_calibration.py",
        "v48_history_calibration.py",
        "crossfit_calibration.py",
        "v50_crossfit_experts.py",
    ]] + [ROOT / "v64_weekday_interaction.py"]
    inputs = [t_path, h_path, ht_path, eh_path, spline_path] + baseline_paths
    if not all(path.exists() for path in inputs + source_paths):
        raise FileNotFoundError("Required frozen V53 inputs or source files are missing")
    frozen = dict(
        benchmark="reused V53 first-future-event development cohort; not an untouched holdout",
        training="All day5 labels only; one first-future event per user",
        evaluation="Day6 first-future-event cohort; labels loaded only after model fit",
        temporal_order="strict day5 timestamps < day6 timestamps",
        baseline="v53_causal_propensity",
        metric="ordinary ROC AUC and evaluation-history-weighted ROC AUC",
        feature_recipe="V53 causal features + smoothed user/item weekday/weekend propensity + UTC day/hour/bucket interactions",
        blend_weight=BLEND_WEIGHT,
        model_params=MODEL_PARAMS,
        selection="Single predeclared CatBoost configuration; no day6 tuning or day6 labels in fit",
        allowed_user_overlap="Historical user overlap is permitted; event IDs and labels are strictly time-separated. Day6 first-future-event labels are never fit targets.",
        inputs={path.name: sha256(path) for path in inputs},
        sources={path.name: sha256(path) for path in source_paths},
    )
    frozen_path = R / "v64_weekday_interaction_corrected_frozen.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        if previous != frozen:
            raise RuntimeError("Frozen V64 manifest differs; do not overwrite or silently reuse this benchmark")
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2))

    with (R / "v48_history_calibrator.pkl").open("rb") as stream:
        import pickle
        spline = pickle.load(stream)["spline"]
    t = np.load(t_path, mmap_mode="r")
    h = np.load(h_path, mmap_mode="r")
    ht = np.load(ht_path, mmap_mode="r")
    evaluation_hist = np.load(eh_path, mmap_mode="r")

    z5, x5, users5, groups5, meta5 = _features(5, t, h, ht, spline)
    base5 = np.load(baseline_paths[0])
    if not np.array_equal(z5["events"], base5["events"]) or not np.array_equal(z5["y"], base5["y"]):
        raise AssertionError("Day5 features do not match frozen V53 events/labels")
    day5_users = _check_first_future_event(5, z5, meta5, t)
    y5 = np.asarray(z5["y"], dtype=np.int8)
    weights5, target_share, source_share5 = _distribution_weights(groups5, evaluation_hist)
    if len(y5) != len(weights5):
        raise AssertionError("Day5 sample weights are not aligned")

    # No day6 arrays, labels, or metrics enter model fitting.
    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(x5, y5, sample_weight=weights5)
    model_path = R / "v64_weekday_interaction_corrected.cbm"
    model.save_model(str(model_path))
    fit_records = dict(
        training_day=5,
        training_rows=len(y5),
        training_events=len(y5) // 10,
        training_users=len(np.unique(day5_users)),
        feature_count=int(x5.shape[1]),
        fit_iterations=int(model.tree_count_),
        train_min_timestamp=int(meta5["rows"][:, 2].min()),
        train_max_timestamp=int(meta5["rows"][:, 2].max()),
        evaluation_history_target_shares=target_share,
        day5_source_history_shares=source_share5,
        training_weekday_shares={DOW_NAMES[i]: float(meta5["weekday_counts"][i] / (len(y5) // 10)) for i in range(7)},
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    del x5, y5, weights5, z5, base5

    z6, x6, users6, groups6, meta6 = _features(6, t, h, ht, spline)
    base6 = np.load(baseline_paths[1])
    if not np.array_equal(z6["events"], base6["events"]) or not np.array_equal(z6["y"], base6["y"]):
        raise AssertionError("Day6 evaluation does not match frozen V53 events/labels")
    day6_users = _check_first_future_event(6, z6, meta6, t)
    event_ids5 = base_paths = np.load(baseline_paths[0])["events"][::10]
    event_ids6 = base6["events"][::10]
    if np.intersect1d(event_ids5, event_ids6).size:
        raise AssertionError("Day5 training and day6 evaluation reuse event IDs")
    if meta5["rows"][:, 2].max() >= meta6["rows"][:, 2].min():
        raise AssertionError("Day5/day6 timestamp boundary is not strictly future")

    candidate_raw = model.predict_proba(x6)[:, 1].astype(np.float32)
    baseline_pred = np.asarray(base6["pred"])
    baseline_rank = rankdata(baseline_pred, method="average") / (len(baseline_pred) + 1.0)
    correction_rank = rankdata(candidate_raw, method="average") / (len(candidate_raw) + 1.0)
    prediction = rankdata((1.0 - BLEND_WEIGHT) * baseline_rank + BLEND_WEIGHT * correction_rank, method="average")
    y6 = np.asarray(z6["y"], dtype=np.int8)
    weights6, _, source_share6 = _distribution_weights(groups6, evaluation_hist)
    baseline_auc = float(roc_auc_score(y6, baseline_pred))
    candidate_auc = float(roc_auc_score(y6, prediction))
    baseline_weighted_auc = _weighted_auc(y6, baseline_pred, weights6)
    candidate_weighted_auc = _weighted_auc(y6, prediction, weights6)
    user_overlap = int(np.intersect1d(day5_users, day6_users).size)
    report = dict(
        status="SCORED_REUSED_DEVELOPMENT_COHORT",
        benchmark=frozen["benchmark"],
        training=fit_records,
        evaluation=dict(
            day=6,
            protocol="first event per user after day6 cutoff; future labels scored only after day5 fit",
            events=len(y6) // 10,
            users=len(np.unique(day6_users)),
            min_timestamp=int(meta6["rows"][:, 2].min()),
            max_timestamp=int(meta6["rows"][:, 2].max()),
            weekday_shares={DOW_NAMES[i]: float(meta6["weekday_counts"][i] / (len(y6) // 10)) for i in range(7)},
            day5_day6_user_overlap=user_overlap,
            event_id_overlap=0,
            source_history_shares=source_share6,
            ordinary_auc=dict(v53=baseline_auc, v64=candidate_auc, gain=candidate_auc - baseline_auc),
            evaluation_history_weighted_auc=dict(v53=baseline_weighted_auc, v64=candidate_weighted_auc, gain=candidate_weighted_auc - baseline_weighted_auc),
        ),
        target_0_85_met=bool(candidate_auc >= 0.85),
        development_gain_positive=bool(candidate_auc > baseline_auc and candidate_weighted_auc > baseline_weighted_auc),
        accepted_as_improvement=False,
        promotion_status="NOT_PROMOTABLE: day5/day6 cohorts were reused and no fresh holdout was evaluated",
        official_auc=None,
        caveats=[
            "This run preserves the previously frozen local day5-to-day6 benchmark; it is not an untouched new-date test because these dates have appeared in earlier repository experiments.",
            "Users can recur across the temporal boundary, but no day6 event IDs or labels enter model fitting; day6 features use only timestamps before the day6 cutoff.",
            "Day5 is Saturday and day6 Sunday in UTC; a single week provides little direct same-weekday history, so explicit weekday features do not eliminate calendar-distribution uncertainty.",
            "Day6 results are reported once under the fixed recipe and are not used to tune this experiment.",
        ],
        model_sha256=sha256(model_path),
        frozen_manifest_sha256=sha256(frozen_path),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    result_path = R / "v64_weekday_interaction_corrected_report.json"
    result_path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        R / "v64_weekday_interaction_corrected_day6.npz",
        y=y6,
        events=base6["events"],
        pred=prediction,
        candidate_raw=candidate_raw,
        baseline=baseline_pred,
    )
    md = [
        "# V64 weekday-interaction correction",
        "",
        "- Protocol: train on day5 only, then evaluate the frozen day6 first-future-event cohort.",
        "- Model: fixed CatBoost nonlinear correction over V53 features plus causal weekday/weekend, hour, bucket, and propensity interactions.",
        f"- Day6 ordinary AUC: V53 `{baseline_auc:.6f}`, V64 `{candidate_auc:.6f}`, gain `{candidate_auc - baseline_auc:+.6f}`.",
        f"- Day6 history-weighted AUC: V53 `{baseline_weighted_auc:.6f}`, V64 `{candidate_weighted_auc:.6f}`, gain `{candidate_weighted_auc - baseline_weighted_auc:+.6f}`.",
        f"- Day6 users: `{len(np.unique(day6_users))}`; day5/day6 user overlap `{user_overlap}`; event-ID overlap `0`.",
        f"- 0.85 reached: `{report['target_0_85_met']}`; official score: none.",
        "",
        "## Validity notes",
        "",
        *[f"- {note}" for note in report["caveats"]],
    ]
    (ROOT / "results" / "v64_weekday_interaction_corrected.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
