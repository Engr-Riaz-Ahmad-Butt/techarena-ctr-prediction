"""Single-recipe, exploratory V68 ALS transfer for Kaggle."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.sparse import coo_matrix
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


RECIPE = {
    "dimension": 128,
    "seed": 68128,
    "iterations": 6,
    "blend_weight": 0.05,
    "click_decay_half_life_days": 3.0,
    "exposure_confidence_scale": 9.0,
    "latent_regularization": 4.0,
    "bias_regularization": 20.0,
    "prior_strength": 50.0,
    "fit_policy": "all labeled events strictly before each validation cutoff",
    "warm_history_policy": "earliest pre-cutoff event per user; unique item-click-timestamp triples",
    "selection_policy": "one fixed recipe; no parameter, seed, epoch, or blend search",
    "day5_gate": {
        "minimum_day6_adjusted_gain": 0.0005,
        "day6_adjusted_95pct_interval_lower_bound": "> 0",
        "minimum_day6_ordinary_gain": 0.0,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_files(root, manifest):
    for name, metadata in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise AssertionError(f"Missing or incomplete input file: {path}")
        if _sha256(path) != metadata["sha256"]:
            raise AssertionError(f"Input checksum mismatch: {path}")


def _verify_bundle(bundle_root):
    manifest_path = bundle_root / "v68_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = bundle_root / "v68_fixed_als_runner.py"
    if _sha256(source_path) != manifest["source_sha256"]:
        raise AssertionError("V68 runner does not match the frozen bundle manifest")
    for name, expected in manifest["initial_factor_sha256"].items():
        if _sha256(bundle_root / name) != expected:
            raise AssertionError(f"Initial factor checksum mismatch: {name}")
    if manifest["recipe"] != RECIPE:
        raise AssertionError("Runner recipe differs from the frozen bundle manifest")
    return manifest


def _rank(values):
    return rankdata(values, method="average") / (len(values) + 1.0)


def _influence(y, prediction):
    positive = np.sort(prediction[y == 1])
    negative = np.sort(prediction[y == 0])
    auc = roc_auc_score(y, prediction)
    out = np.empty(len(y), dtype=np.float64)
    mask = y == 1
    out[mask] = (
        (
            np.searchsorted(negative, prediction[mask], "left")
            + np.searchsorted(negative, prediction[mask], "right")
        )
        / (2 * len(negative))
        - auc
    ) / len(positive)
    out[~mask] = (
        1
        - (
            np.searchsorted(positive, prediction[~mask], "left")
            + np.searchsorted(positive, prediction[~mask], "right")
        )
        / (2 * len(positive))
        - auc
    ) / len(negative)
    return out.reshape(-1, 10).sum(axis=1)


def _weighted_influence(y, prediction, weights):
    def cdf(values, sample_weights, query):
        order = np.argsort(values)
        values = values[order]
        cumulative = np.r_[0.0, np.cumsum(sample_weights[order])]
        left = np.searchsorted(values, query, "left")
        right = np.searchsorted(values, query, "right")
        return (cumulative[left] + cumulative[right]) / (2 * sample_weights.sum())

    positive = y == 1
    auc = roc_auc_score(y, prediction, sample_weight=weights)
    out = np.empty(len(y), dtype=np.float64)
    out[positive] = (
        weights[positive]
        / weights[positive].sum()
        * (cdf(prediction[~positive], weights[~positive], prediction[positive]) - auc)
    )
    out[~positive] = (
        weights[~positive]
        / weights[~positive].sum()
        * (
            1
            - cdf(prediction[positive], weights[positive], prediction[~positive])
            - auc
        )
    )
    return out.reshape(-1, 10).sum(axis=1)


@njit(parallel=True)
def _solve_rows(
    indptr, indices, weights, targets, factors, other_bias, gram, rhs,
    background, global_rate, prior_bias, latent_reg, bias_reg,
):
    count = len(indptr) - 1
    dimension = factors.shape[1]
    biases = np.empty(count, dtype=np.float64)
    latent = np.empty((count, dimension), dtype=np.float64)
    for row in prange(count):
        matrix = background[row] * gram.copy()
        vector_rhs = background[row] * rhs.copy()
        matrix[0, 0] += bias_reg
        vector_rhs[0] += bias_reg * prior_bias[row]
        for j in range(1, dimension + 1):
            matrix[j, j] += latent_reg
        vector = np.empty(dimension + 1, dtype=np.float64)
        vector[0] = 1.0
        for offset in range(indptr[row], indptr[row + 1]):
            item = indices[offset]
            weight = weights[offset]
            for j in range(dimension):
                vector[j + 1] = factors[item, j]
            residual = targets[offset] - weight * (global_rate + other_bias[item])
            for j in range(dimension + 1):
                vector_rhs[j] += residual * vector[j]
                for k in range(dimension + 1):
                    matrix[j, k] += weight * vector[j] * vector[k]
        solution = np.linalg.solve(matrix, vector_rhs)
        biases[row] = solution[0]
        latent[row] = solution[1:]
    return biases, latent


def _paired_matrices(users, items, weights, targets, shape):
    exposure = coo_matrix((weights, (users, items)), shape=shape).tocsr()
    positive = coo_matrix((targets, (users, items)), shape=shape).tocsr()
    if not np.array_equal(exposure.indptr, positive.indptr):
        raise AssertionError("Exposure and positive CSR row pointers differ")
    if not np.array_equal(exposure.indices, positive.indices):
        raise AssertionError("Exposure and positive CSR column indices differ")
    return exposure, positive


def _warm_history(data, fitting_indices, cutoff):
    timestamps = data["timestamp"][fitting_indices]
    users = data["users"][fitting_indices]
    order = np.argsort(timestamps, kind="stable")
    sorted_users = users[order]
    _, first = np.unique(sorted_users, return_index=True)
    first_events = fitting_indices[order[np.sort(first)]]
    history_items = data["hist"][first_events]
    gaps = data["gaps"][first_events]
    valid = history_items >= 0
    local_users = np.repeat(data["users"][first_events], history_items.shape[1])[valid.ravel()]
    items = history_items.ravel()[valid.ravel()].astype(np.int32, copy=False)
    event_time = np.repeat(data["timestamp"][first_events], history_items.shape[1])[valid.ravel()]
    gap_days = gaps.ravel()[valid.ravel()].astype(np.float64)
    history_time = np.rint(event_time - gap_days * 86400.0).astype(np.int64)
    if np.any(history_time >= np.repeat(data["timestamp"][first_events], history_items.shape[1])[valid.ravel()]):
        raise AssertionError("A warm-history click is not strictly before its event")
    keys = np.rec.fromarrays(
        [local_users.astype(np.int32), items, history_time],
        names="user,item,timestamp",
    )
    _, unique = np.unique(keys, return_index=True)
    local_users = local_users[unique].astype(np.int32, copy=False)
    items = items[unique].astype(np.int32, copy=False)
    history_time = history_time[unique]
    ages = np.maximum((cutoff - history_time) / 86400.0, 0.0)
    decay = np.exp2(-ages / RECIPE["click_decay_half_life_days"]).astype(np.float32)
    return local_users, items, decay


def _initial_item_factors(bundle_root, day, item_count):
    path = bundle_root / f"imp15_model_day{day}.npz"
    with np.load(path) as archive:
        values = archive["V"].astype(np.float64)
    if values.shape != (item_count, 16):
        raise AssertionError(f"Unexpected V68 initialization shape: {values.shape}")
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(RECIPE["seed"])
    extra = rng.normal(0.0, 0.05, size=(item_count, RECIPE["dimension"] - values.shape[1]))
    factors = np.column_stack((values, extra))
    factors /= np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-12)
    return factors


def _fit(day, data, validation, start, bundle_root, output):
    cutoff = start + day * 86400
    all_times = data["timestamp"]
    fitting_indices = np.flatnonzero(all_times < cutoff)
    if not len(fitting_indices):
        raise AssertionError("No strictly pre-cutoff fitting events")
    if int(all_times[fitting_indices].max()) >= cutoff:
        raise AssertionError("Fitting events cross the validation cutoff")

    target = data["target"][fitting_indices].astype(np.int64)
    users_by_event = data["users"][fitting_indices].astype(np.int32)
    items_by_event = data["items"][fitting_indices].astype(np.int32)
    event_times = all_times[fitting_indices].astype(np.int64)
    labels = np.zeros(items_by_event.shape, dtype=np.float32)
    clicked = target < 10
    labels[np.flatnonzero(clicked), target[clicked]] = 1.0
    age = np.maximum((cutoff - event_times) / 86400.0, 0.0)
    event_decay = np.exp2(-age / RECIPE["click_decay_half_life_days"]).astype(np.float32)
    impression_decay = np.repeat(event_decay, 10)
    flat_users = np.repeat(users_by_event, 10).astype(np.int32, copy=False)
    flat_items = items_by_event.reshape(-1).astype(np.int32, copy=False)
    flat_labels = labels.reshape(-1)
    flat_targets = flat_labels * impression_decay
    global_rate = float(flat_targets.sum() / impression_decay.sum())

    user_count = int(data["users"].max()) + 1
    item_count = int(data["items"].max()) + 1
    user_exposure = np.bincount(flat_users, weights=impression_decay, minlength=user_count)
    item_exposure = np.bincount(flat_items, weights=impression_decay, minlength=item_count)
    user_clicks = np.bincount(flat_users, weights=flat_targets, minlength=user_count)
    item_clicks = np.bincount(flat_items, weights=flat_targets, minlength=item_count)
    prior = RECIPE["prior_strength"]
    user_prior = (user_clicks + prior * global_rate) / (user_exposure + prior) - global_rate
    item_prior = (item_clicks + prior * global_rate) / (item_exposure + prior) - global_rate

    warm_users, warm_items, warm_decay = _warm_history(data, fitting_indices, cutoff)
    confidence = (
        RECIPE["exposure_confidence_scale"]
        * np.bincount(warm_users, weights=warm_decay, minlength=user_count)
        / item_count
    )
    matrix_users = np.concatenate((flat_users, warm_users))
    matrix_items = np.concatenate((flat_items, warm_items))
    matrix_weights = np.concatenate((impression_decay, warm_decay)).astype(np.float32, copy=False)
    matrix_targets = np.concatenate((flat_targets, warm_decay)).astype(np.float32, copy=False)
    exposure_matrix, target_matrix = _paired_matrices(
        matrix_users, matrix_items, matrix_weights, matrix_targets, (user_count, item_count)
    )
    item_exposure_matrix = exposure_matrix.T.tocsr()
    item_target_matrix = target_matrix.T.tocsr()
    if not np.array_equal(item_exposure_matrix.indptr, item_target_matrix.indptr):
        raise AssertionError("Transposed exposure and target row pointers differ")
    if not np.array_equal(item_exposure_matrix.indices, item_target_matrix.indices):
        raise AssertionError("Transposed exposure and target indices differ")

    factors = _initial_item_factors(bundle_root, day, item_count)
    user_factors = np.zeros((user_count, RECIPE["dimension"]), dtype=np.float64)
    user_bias = user_prior.copy()
    item_bias = item_prior.copy()
    started = time.monotonic()
    latent_reg = RECIPE["latent_regularization"]
    bias_reg = RECIPE["bias_regularization"]
    for iteration in range(1, RECIPE["iterations"] + 1):
        augmented = np.column_stack((np.ones(item_count), factors))
        user_bias, user_factors = _solve_rows(
            exposure_matrix.indptr, exposure_matrix.indices, exposure_matrix.data,
            target_matrix.data, factors, item_bias, augmented.T @ augmented,
            -augmented.T @ (global_rate + item_bias), confidence, global_rate,
            user_prior, latent_reg, bias_reg,
        )
        augmented_users = np.column_stack((np.ones(user_count), user_factors))
        item_bias, factors = _solve_rows(
            item_exposure_matrix.indptr, item_exposure_matrix.indices,
            item_exposure_matrix.data, item_target_matrix.data, user_factors,
            user_bias, augmented_users.T @ (confidence[:, None] * augmented_users),
            -augmented_users.T @ (confidence * (global_rate + user_bias)),
            np.ones(item_count), global_rate, item_prior, latent_reg, bias_reg,
        )
        if not all(np.isfinite(array).all() for array in (user_factors, factors, user_bias, item_bias)):
            raise FloatingPointError(f"Nonfinite ALS parameters after iteration {iteration}")
        print(
            json.dumps({
                "day": day,
                "iteration": iteration,
                "iterations": RECIPE["iterations"],
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }),
            flush=True,
        )

    validation_indices = validation["indices"].astype(np.int64)
    if not np.all(all_times[validation_indices] >= cutoff):
        raise AssertionError("Validation event precedes its cutoff")
    validation_targets = data["target"][validation_indices]
    validation_labels = np.zeros((len(validation_indices), 10), dtype=np.float32)
    selected = validation_targets < 10
    validation_labels[np.flatnonzero(selected), validation_targets[selected]] = 1.0
    if not np.array_equal(validation_labels.reshape(-1), validation["y"]):
        raise AssertionError(f"Day {day} validation labels are misaligned")
    if len(np.unique(validation_indices)) != len(validation_indices):
        raise AssertionError("Validation event indices must be unique")

    validation_users = data["users"][validation_indices].astype(np.int32)
    validation_items = data["items"][validation_indices].astype(np.int32)
    history_items = data["hist"][validation_indices]
    history_gaps = data["gaps"][validation_indices].astype(np.float64)
    if np.any((history_items >= 0) & (history_gaps <= 0)):
        raise AssertionError("Validation history contains a non-prior click")
    event_exposed = np.bincount(flat_users, minlength=user_count)
    cold = event_exposed[validation_users] == 0
    score_users = user_factors[validation_users].copy()
    score_bias = user_bias[validation_users].copy()
    foldin = None
    if cold.any():
        cold_events = np.flatnonzero(cold)
        h = history_items[cold]
        valid = h >= 0
        cold_users = np.repeat(validation_users[cold], h.shape[1])[valid.ravel()]
        cold_items = h.ravel()[valid.ravel()].astype(np.int32)
        ages = history_gaps[cold][valid]  # expressed in days before each event
        cold_decay = np.exp2(-ages / RECIPE["click_decay_half_life_days"]).astype(np.float32)
        local_rows = np.repeat(np.arange(len(cold_events), dtype=np.int32), h.shape[1])[valid.ravel()]
        user_matrix, target_matrix_foldin = _paired_matrices(
            local_rows, cold_items, cold_decay, cold_decay, (len(cold_events), item_count)
        )
        augmented_items = np.column_stack((np.ones(item_count), factors))
        cold_confidence = (
            RECIPE["exposure_confidence_scale"]
            * np.bincount(local_rows, weights=cold_decay, minlength=len(cold_events))
            / item_count
        )
        folded_bias, folded_users = _solve_rows(
            user_matrix.indptr, user_matrix.indices, user_matrix.data,
            target_matrix_foldin.data, factors, item_bias,
            augmented_items.T @ augmented_items,
            -augmented_items.T @ (global_rate + item_bias),
            cold_confidence,
            global_rate, np.zeros(len(cold_events)), latent_reg, bias_reg,
        )
        foldin = (
            user_matrix, target_matrix_foldin, cold_confidence, augmented_items,
            local_rows, cold_decay, cold_events,
        )
        score_users[cold] = folded_users
        score_bias[cold] = folded_bias
    raw = (
        global_rate
        + score_bias[:, None]
        + item_bias[validation_items]
        + np.einsum("ed,ecd->ec", score_users, factors[validation_items])
    ).reshape(-1)
    if not np.isfinite(raw).all():
        raise FloatingPointError("ALS predictions contain nonfinite values")
    prediction = (
        (1.0 - RECIPE["blend_weight"]) * _rank(validation["baseline"])
        + RECIPE["blend_weight"] * _rank(raw)
    )
    if not ((prediction >= 0).all() and (prediction <= 1).all()):
        raise AssertionError("Rank blend is outside [0,1]")

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "model.npz"
    np.savez_compressed(
        checkpoint,
        U=user_factors,
        V=factors,
        bu=user_bias,
        bi=item_bias,
        g=np.float64(global_rate),
        exposures=event_exposed,
        cold_validation_events=cold,
    )
    with np.load(checkpoint) as saved:
        restored = {
            "U": saved["U"],
            "V": saved["V"],
            "bu": saved["bu"],
            "bi": saved["bi"],
            "g": float(saved["g"]),
        }
    replay_users = restored["U"][validation_users].copy()
    replay_bias = restored["bu"][validation_users].copy()
    if foldin is not None:
        user_matrix, target_matrix_foldin, cold_confidence, _, _, _, cold_events = foldin
        replay_augmented_items = np.column_stack((np.ones(item_count), restored["V"]))
        replay_bias_cold, replay_users_cold = _solve_rows(
            user_matrix.indptr, user_matrix.indices, user_matrix.data,
            target_matrix_foldin.data, restored["V"], restored["bi"],
            replay_augmented_items.T @ replay_augmented_items,
            -(replay_augmented_items.T @ (restored["g"] + restored["bi"])),
            cold_confidence, restored["g"], np.zeros(len(cold_events)),
            latent_reg, bias_reg,
        )
        replay_users[cold] = replay_users_cold
        replay_bias[cold] = replay_bias_cold
    replay = (
        restored["g"]
        + replay_bias[:, None]
        + restored["bi"][validation_items]
        + np.einsum("ed,ecd->ec", replay_users, restored["V"][validation_items])
    ).reshape(-1)
    replay_error = float(np.max(np.abs(raw - replay)))
    if replay_error != 0:
        raise AssertionError(f"Checkpoint replay differs by {replay_error}")
    replay_pred = (
        (1.0 - RECIPE["blend_weight"]) * _rank(validation["baseline"])
        + RECIPE["blend_weight"] * _rank(replay)
    )
    if not np.array_equal(replay_pred, prediction):
        raise AssertionError("Rank-blended checkpoint replay differs")

    y = validation["y"]
    weights = validation["weights"]
    baseline = validation["baseline"]
    ordinary_auc = float(roc_auc_score(y, prediction))
    baseline_auc = float(roc_auc_score(y, baseline))
    adjusted_auc = float(roc_auc_score(y, prediction, sample_weight=weights))
    baseline_adjusted_auc = float(roc_auc_score(y, baseline, sample_weight=weights))
    ordinary_gain = ordinary_auc - baseline_auc
    adjusted_gain = adjusted_auc - baseline_adjusted_auc
    ordinary_se = float(np.sqrt(np.sum((_influence(y, prediction) - _influence(y, baseline)) ** 2)))
    adjusted_se = float(np.sqrt(np.sum(
        (_weighted_influence(y, prediction, weights) - _weighted_influence(y, baseline, weights)) ** 2
    )))
    report = {
        "day": day,
        "exploratory_only": True,
        "rows": len(y),
        "events": len(y) // 10,
        "fit_events": len(fitting_indices),
        "fit_min_timestamp": int(all_times[fitting_indices].min()),
        "fit_max_timestamp": int(all_times[fitting_indices].max()),
        "validation_min_timestamp": int(all_times[validation_indices].min()),
        "ordinary_auc": ordinary_auc,
        "baseline_ordinary_auc": baseline_auc,
        "ordinary_gain": ordinary_gain,
        "ordinary_gain_95pct_event_interval": [
            ordinary_gain - 1.96 * ordinary_se,
            ordinary_gain + 1.96 * ordinary_se,
        ],
        "history_adjusted_auc": adjusted_auc,
        "baseline_history_adjusted_auc": baseline_adjusted_auc,
        "adjusted_gain": adjusted_gain,
        "adjusted_gain_95pct_event_interval": [
            adjusted_gain - 1.96 * adjusted_se,
            adjusted_gain + 1.96 * adjusted_se,
        ],
        "raw_auc": float(roc_auc_score(y, raw)),
        "saved_model_replay": "PASS",
        "saved_model_replay_max_error": replay_error,
        "model_sha256": _sha256(checkpoint),
        "training_seconds": round(time.monotonic() - started, 1),
        "csv_created": False,
        "official_auc": None,
    }
    np.savez_compressed(
        output / "predictions.npz",
        raw=raw,
        pred=prediction,
        baseline=baseline,
        y=y,
        weights=weights,
        event_indices=validation_indices,
        event_ids=np.repeat(validation_indices, 10),
        cold_validation_events=cold,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def _device_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        gpus = []
    return {
        "kaggle_gpu_devices": gpus,
        "als_compute": "CPU Numba parallel row solves; GPU is recorded but not used by this solver",
    }


def run(bundle_root, neural_root, output_root):
    bundle_root = Path(bundle_root).resolve()
    neural_root = Path(neural_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    bundle_manifest = _verify_bundle(bundle_root)
    neural_manifest_path = neural_root / "neural_manifest.json"
    neural_manifest = json.loads(neural_manifest_path.read_text())
    _verify_files(neural_root, neural_manifest)
    source_hash = _sha256(bundle_root / "v68_fixed_als_runner.py")
    device = _device_info()
    with np.load(neural_root / "events.npz") as archive:
        data = {key: archive[key].copy() for key in archive.files}
    start = int(neural_manifest["start"])
    if not np.all(np.diff(data["timestamp"]) >= 0):
        raise AssertionError("Training event timestamps must be chronological")
    if not (len(data["users"]) == len(data["timestamp"]) == len(data["target"])):
        raise AssertionError("Event arrays have different lengths")
    if data["items"].shape != (len(data["users"]), 10) or data["hist"].shape != (len(data["users"]), 20):
        raise AssertionError("Unexpected event or history dimensions")

    set_num_threads(4)
    provenance = {
        "experiment": "ctr-v68-fixed-decay-001",
        "recipe": RECIPE,
        "runner_sha256": source_hash,
        "bundle_manifest_sha256": _sha256(bundle_root / "v68_bundle_manifest.json"),
        "initial_factor_sha256": bundle_manifest["initial_factor_sha256"],
        "neural_manifest_sha256": _sha256(neural_manifest_path),
        "neural_input_sha256": {
            name: metadata["sha256"] for name, metadata in neural_manifest["files"].items()
        },
        "device": device,
        "exploratory_only": True,
        "submission_created": False,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    reports = []
    for day in (6, 5):
        with np.load(neural_root / f"val{day}.npz") as archive:
            validation = {key: archive[key].copy() for key in archive.files}
        reports.append(_fit(day, data, validation, start, bundle_root, output_root / f"day{day}"))
        if day == 6:
            gate_passed = (
                reports[-1]["adjusted_gain"] >= RECIPE["day5_gate"]["minimum_day6_adjusted_gain"]
                and reports[-1]["adjusted_gain_95pct_event_interval"][0] > 0
                and reports[-1]["ordinary_gain"] >= RECIPE["day5_gate"]["minimum_day6_ordinary_gain"]
            )
            if not gate_passed:
                break

    summary = {
        "experiment": "ctr-v68-fixed-decay-001",
        "platform": "Kaggle",
        "exploratory_only": True,
        "promotion": False,
        "submission_created": False,
        "official_auc": None,
        "target_official_auc": 0.82,
        "official_target_achieved": None,
        "recipe": RECIPE,
        "recipe_source": "Fixed before reading new day5/day6 metrics; no search or tuning",
        "device": device,
        "runner_sha256": source_hash,
        "bundle_manifest_sha256": _sha256(bundle_root / "v68_bundle_manifest.json"),
        "neural_manifest_sha256": _sha256(neural_manifest_path),
        "input_hashes": {
            "initial_factors": bundle_manifest["initial_factor_sha256"],
            "neural_inputs": provenance["neural_input_sha256"],
        },
        "day_reports": reports,
        "day5_run": len(reports) == 2,
        "day5_skipped_gate_failure": len(reports) == 1,
        "note": (
            "Day5/day6 are reused development cohorts. Results are exploratory only, "
            "cannot support promotion, and are not an official leaderboard estimate."
        ),
    }
    (output_root / "run_config.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
