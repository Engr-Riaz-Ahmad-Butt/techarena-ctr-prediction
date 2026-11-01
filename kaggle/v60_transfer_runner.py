"""Transfer the frozen V60 XGBoost expert to the main causal validation dates."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

PARAMS = dict(
    objective="binary:logistic",
    tree_method="hist",
    max_bin=63,
    max_depth=6,
    eta=0.05,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=5.0,
    reg_alpha=0.1,
    nthread=4,
    seed=6001,
    max_cached_hist_node=32,
)
ROUNDS = 700
BLEND_WEIGHT = 0.1


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict(model, features, cp):
    chunks = []
    for offset in range(0, len(features), 25000):
        x = features.block(offset, min(offset + 25000, len(features)))
        prediction = model.inplace_predict(cp.asarray(x))
        chunks.append(cp.asnumpy(prediction))
    result = np.concatenate(chunks)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise AssertionError("XGBoost returned invalid probabilities")
    return result


def run(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if tuple(int(part) for part in xgb.__version__.split(".")[:2]) < (3, 0):
        raise RuntimeError("XGBoost >=3.0 is required")
    h = _load_module("v60_transfer_feature_helpers", root / "runner.py")
    device, cp = h.choose_device("cuda")
    if device != "cuda" or cp is None:
        raise RuntimeError("This transfer experiment requires Kaggle CUDA")
    output.mkdir(parents=True)
    manifest = h.load_bundle(root, verify=True)
    days = [2, 3]
    specs = [manifest["train"][str(day)] for day in days]
    labels = []
    for spec in specs:
        metadata = json.loads((root / spec["metadata"]).read_text())
        labels.append(np.load(root / spec["labels"]))
        if int(metadata["last_timestamp"]) >= int(manifest["validation"]["6"]["cutoff"]):
            raise AssertionError("Training labels cross the day-6 cutoff")
    y = np.concatenate(labels).astype(np.float32)
    if len(y) % 10 or ((y.reshape(-1, 10).sum(1) > 1).any()):
        raise AssertionError("Training labels must form valid ten-candidate events")
    features = h.FeatureBlocks(root, [spec["features"] for spec in specs])
    if len(features) != len(y):
        raise AssertionError("Training feature/label row counts differ")

    frozen = dict(
        experiment="ctr-v60-transfer-001",
        training_days=days,
        training_rows=len(y),
        training_positive_rows=int(y.sum()),
        validation_days=[6, 5],
        rounds=ROUNDS,
        params=PARAMS,
        blend_weight=BLEND_WEIGHT,
        device=device,
        xgboost=xgb.__version__,
        feature_manifest_sha256=_digest(root / "bundle_manifest.json"),
        source_sha256=_digest(Path(__file__)),
        selection="Day 6 first; day 5 only if day-6 confidence gate passes; no tuning.",
        validation_labels_used_for_fitting=False,
        submission_created=False,
    )
    (output / "recipe.json").write_text(json.dumps(frozen, indent=2))

    iterator = h.BatchIterator(features, y, cp=cp, batch_size=100000)
    matrix = xgb.QuantileDMatrix(iterator, nthread=4, max_bin=63)
    if matrix.num_row() != len(y) or matrix.num_col() != 354:
        raise AssertionError("Quantized training matrix has the wrong dimensions")
    started = time.monotonic()
    model = xgb.train(dict(PARAMS, device=device), matrix, num_boost_round=ROUNDS)
    model_path = output / "model.ubj"
    model.save_model(model_path)
    del matrix, iterator

    reports = []
    for day in (6, 5):
        validation = manifest["validation"][str(day)]
        with np.load(root / validation["data"]) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        validation_features = h.FeatureBlocks(root, [validation["features"]])
        if len(validation_features) != len(values["y"]):
            raise AssertionError(f"Day {day} validation feature/label rows differ")
        raw = _predict(model, validation_features, cp)
        if len(raw) != len(values["y"]):
            raise AssertionError(f"Day {day} prediction rows differ from labels")
        prediction = h.rank(
            (1 - BLEND_WEIGHT) * h.rank(values["baseline"])
            + BLEND_WEIGHT * h.rank(raw)
        )
        report = h.metrics(values["y"], prediction, values["baseline"], values["weights"])

        restored = xgb.Booster()
        restored.load_model(model_path)
        restored.set_param({"device": device, "nthread": 4})
        replay = _predict(restored, validation_features, cp)
        replay_error = float(np.max(np.abs(raw - replay)))
        if replay_error != 0:
            raise AssertionError(f"Saved-model replay changed by {replay_error}")
        report.update(
            day=day,
            raw_auc=float(h.roc_auc_score(values["y"], raw)),
            rows=len(raw),
            saved_checkpoint_replay="PASS",
            saved_model_replay_max_error=replay_error,
            model_sha256=_digest(model_path),
            elapsed_seconds=round(time.monotonic() - started, 1),
            official_auc=None,
            csv_ready=False,
        )
        np.savez_compressed(
            output / f"day{day}_predictions.npz",
            raw=raw,
            pred=prediction,
            **values,
        )
        (output / f"day{day}_report.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
        if day == 6 and (
            report["adjusted_gain"] <= 0
            or report["adjusted_gain_interval"][0] <= 0
            or report["ordinary_gain"] <= 0
        ):
            break

    eligible = len(reports) == 2 and all(report["positive_gate"] for report in reports)
    summary = dict(
        experiment="ctr-v60-transfer-001",
        reports=reports,
        eligible_for_review=eligible,
        official_auc=None,
        target_official_auc=0.82,
        official_target_achieved=None,
        csv_ready=False,
        submission_created=False,
        note="Development validation only; both dates have prior exposure and cannot establish official leaderboard performance.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
