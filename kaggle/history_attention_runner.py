"""Fit a frozen-parent candidate-specific history-attention residual on Kaggle.

The candidate is exploratory on repeatedly reused temporal cohorts. It does not
create a competition submission or estimate an official leaderboard score.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from history_attention_model import HistoryChoice
from tabular_choice_setwise_runner import ChoiceNet, RECIPE as PARENT_RECIPE

DAY = 86400
ATTENTION_RECIPE = dict(
    epochs=6,
    batch_size=512,
    learning_rate=0.0005,
    weight_decay=0.01,
    residual_penalty=0.05,
    seed=6902,
    blend_weight=0.1,
    attention_hidden=32,
    residual_hidden=64,
    maximum_logit_delta=0.5,
    fit_days=3,
    parent_epochs=8,
    parent_seed=6802,
)


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


def _target_from_labels(labels):
    values = np.asarray(labels).reshape(-1, 10)
    if not np.isin(values, [0, 1]).all() or (values.sum(1) > 1).any():
        raise ValueError("Expected binary labels with at most one click per event")
    return np.where(values.sum(1) > 0, values.argmax(1), 10).astype(np.int64)


def _aligned_events(neural_data, manifest, day, labels):
    cutoff = int(manifest["start"]) + day * DAY
    lower = cutoff - ATTENTION_RECIPE["fit_days"] * DAY
    selected = np.flatnonzero(
        (neural_data["timestamp"] >= lower)
        & (neural_data["timestamp"] < cutoff)
    )
    targets = _target_from_labels(labels)
    if len(selected) != len(targets):
        raise AssertionError(
            f"Fitting event count mismatch for day {day}: "
            f"{len(selected)} != {len(targets)}"
        )
    if len(selected) and not np.array_equal(
        selected, np.arange(selected[0], selected[-1] + 1)
    ):
        raise AssertionError("Fitting events must be contiguous in history order")
    if not np.all(np.diff(neural_data["timestamp"][selected]) >= 0):
        raise AssertionError("History events must be ordered chronologically")
    if not np.array_equal(neural_data["target"][selected], targets):
        raise AssertionError(f"Feature labels and event targets differ for day {day}")
    if not (neural_data["timestamp"][selected] < cutoff).all():
        raise AssertionError("Fitting events must precede the validation cutoff")
    return selected, targets


def _prediction(model, features, event_data, event_ids, device, batch_size=1024):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(event_ids), batch_size):
            ids = event_ids[start : start + batch_size]
            row_start = start * 10
            row_end = (start + len(ids)) * 10
            x = features.block(row_start, row_end).reshape(-1, 10, 354)
            x = torch.as_tensor(x, device=device)
            if isinstance(model, HistoryChoice):
                tensors = (
                    x,
                    torch.as_tensor(event_data["items"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["hist"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["gaps"][ids], device=device, dtype=torch.float32),
                    torch.as_tensor(event_data["bucket"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["kind"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["timestamp"][ids], device=device, dtype=torch.long),
                )
                logits, _ = model(*tensors)
            else:
                logits = model(x)
            chunks.append(logits.softmax(1)[:, :10].cpu().numpy())
    return np.concatenate(chunks).reshape(-1)


def _train_parent(root, neural_root, day, manifest, weighting, h, device, output):
    specs = [manifest["train"][str(d)] for d in range(day - 3, day)]
    validation = manifest["validation"][str(day)]
    label_parts = []
    for spec in specs:
        metadata = json.loads((root / spec["metadata"]).read_text())
        if int(metadata["last_timestamp"]) >= int(validation["cutoff"]):
            raise AssertionError("Parent fitting labels cross the validation cutoff")
        label_parts.append(np.load(root / spec["labels"]))
    labels = np.concatenate(label_parts).astype(np.float32)
    if len(labels) % 10:
        raise AssertionError("Parent fitting rows must contain complete events")
    target_np = _target_from_labels(labels)
    target = torch.as_tensor(target_np, device=device, dtype=torch.long)
    weights_np, weighting_recipe = weighting.training_weights(
        neural_root, day, labels, h
    )
    if len(weights_np) != len(labels):
        raise AssertionError("Parent row weights are not aligned with labels")
    weights_np = np.asarray(weights_np).reshape(-1, 10)
    if not np.array_equal(weights_np, np.repeat(weights_np[:, :1], 10, axis=1)):
        raise AssertionError("History-importance weights must be event-constant")
    weights = torch.as_tensor(weights_np[:, 0].copy(), device=device)

    features = h.FeatureBlocks(root, [spec["features"] for spec in specs])
    if len(features) != len(labels):
        raise AssertionError("Parent feature/label row counts differ")
    x = torch.empty((len(labels), 354), device=device, dtype=torch.float32)
    for offset in range(0, len(labels), 50000):
        end = min(offset + 50000, len(labels))
        x[offset:end] = torch.as_tensor(
            features.block(offset, end), device=device, dtype=torch.float32
        )
    variance, mean = torch.var_mean(x, dim=0, unbiased=False)
    scale = variance.sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise AssertionError("Non-finite parent feature normalization")
    x = x.reshape(-1, 10, 354)

    torch.manual_seed(ATTENTION_RECIPE["parent_seed"])
    model = ChoiceNet(mean, scale, PARENT_RECIPE).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PARENT_RECIPE["learning_rate"],
        weight_decay=PARENT_RECIPE["weight_decay"],
    )
    for epoch in range(PARENT_RECIPE["epochs"]):
        model.train()
        order = torch.randperm(len(x), device=device)
        loss_total = weight_total = 0.0
        for offset in range(0, len(x), PARENT_RECIPE["batch_size"]):
            ix = order[offset : offset + PARENT_RECIPE["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.cross_entropy(
                model(x[ix]), target[ix], reduction="none"
            )
            loss = (losses * weights[ix]).sum() / weights[ix].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float((losses.detach() * weights[ix]).sum())
            weight_total += float(weights[ix].sum())
        print(
            {
                "stage": "parent_fit",
                "day": day,
                "epoch": epoch + 1,
                "epochs": PARENT_RECIPE["epochs"],
                "weighted_loss": loss_total / weight_total,
            },
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "parent.pt"
    torch.save(model.state_dict(), checkpoint)
    parent_metadata = {
        "day": day,
        "fit_rows": len(labels),
        "fit_events": len(target_np),
        "fit_days": list(range(day - 3, day)),
        "validation_cutoff": int(validation["cutoff"]),
        "parent_recipe": PARENT_RECIPE,
        "weighting": weighting_recipe,
        "checkpoint_sha256": _digest(checkpoint),
    }
    (output / "parent_recipe.json").write_text(json.dumps(parent_metadata, indent=2))
    del x, target, weights, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return model, features, target_np, weights_np[:, 0], parent_metadata


def _load_validation(root, validation, h):
    with np.load(root / validation["data"]) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if len(values["y"]) % 10 or len(values["weights"]) != len(values["y"]):
        raise AssertionError("Malformed validation event arrays")
    return values, h.FeatureBlocks(root, [validation["features"]])


def _train_attention(
    root, day, model, features, event_data, fitting_ids, targets, event_weights,
    neural_root, validation, h, device, output,
):
    with np.load(neural_root / f"factors{day}.npz") as factor_archive:
        item_vectors = factor_archive["V"].astype(np.float32)
        history_vectors = factor_archive["H"].astype(np.float32)
    if item_vectors.shape[0] <= int(event_data["items"].max()):
        raise AssertionError("Item factor matrix does not cover candidate IDs")
    if history_vectors.shape[0] <= int(event_data["hist"].max()):
        raise AssertionError("History factor matrix does not cover history IDs")

    attention = HistoryChoice(
        model, item_vectors, history_vectors,
        scale=ATTENTION_RECIPE["maximum_logit_delta"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in attention.parameters() if parameter.requires_grad],
        lr=ATTENTION_RECIPE["learning_rate"],
        weight_decay=ATTENTION_RECIPE["weight_decay"],
    )
    rng = np.random.default_rng(ATTENTION_RECIPE["seed"])
    output.mkdir(parents=True, exist_ok=False)
    recipe = dict(
        experiment="ctr-history-attention-001",
        day=day,
        recipe=ATTENTION_RECIPE,
        parent_recipe=PARENT_RECIPE,
        fit_events=len(fitting_ids),
        fit_first_timestamp=int(event_data["timestamp"][fitting_ids].min()),
        fit_last_timestamp=int(event_data["timestamp"][fitting_ids].max()),
        validation_cutoff=int(validation["cutoff"]),
        feature_manifest_sha256=h.digest(neural_root / "neural_manifest.json"),
        source_sha256=_digest(Path(__file__)),
        torch_version=torch.__version__,
        parent_checkpoint_sha256=_digest(output.parent / "parent" / "parent.pt"),
        feature_transform=PARENT_RECIPE.get("feature_transform"),
        validation_labels_used_for_optimization=False,
    )
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))

    for epoch in range(ATTENTION_RECIPE["epochs"]):
        attention.train()
        if not np.array_equal(fitting_ids, np.arange(fitting_ids[0], fitting_ids[-1] + 1)):
            raise AssertionError("Training feature events must form a contiguous time slice")
        n_events = len(fitting_ids)
        block_size = 2048
        blocks = np.arange(0, n_events, block_size)
        total_loss = event_count = 0.0
        for block_start in rng.permutation(blocks):
            block_end = min(int(block_start) + block_size, n_events)
            ids_in_block = fitting_ids[block_start:block_end]
            local_x = features.block(int(block_start) * 10, block_end * 10)
            local_x = local_x.reshape(-1, 10, 354)
            order = rng.permutation(len(ids_in_block))
            for offset in range(0, len(order), ATTENTION_RECIPE["batch_size"]):
                local_ix = order[offset : offset + ATTENTION_RECIPE["batch_size"]]
                ids = ids_in_block[local_ix]
                args = (
                    torch.as_tensor(local_x[local_ix], device=device),
                    torch.as_tensor(event_data["items"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["hist"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["gaps"][ids], device=device, dtype=torch.float32),
                    torch.as_tensor(event_data["bucket"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["kind"][ids], device=device, dtype=torch.long),
                    torch.as_tensor(event_data["timestamp"][ids], device=device, dtype=torch.long),
                )
                relative_ids = block_start + local_ix
                y = torch.as_tensor(targets[relative_ids], device=device, dtype=torch.long)
                w = torch.as_tensor(
                    event_weights[relative_ids], device=device, dtype=torch.float32
                )
                optimizer.zero_grad(set_to_none=True)
                logits, delta = attention(*args)
                losses = nn.functional.cross_entropy(logits, y, reduction="none")
                loss = (losses * w).sum() / w.sum()
                loss = loss + ATTENTION_RECIPE["residual_penalty"] * delta.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite history-attention loss")
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in attention.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
                total_loss += float(loss.detach()) * len(ids)
                event_count += len(ids)
        print(
            {
                "stage": "attention_fit",
                "day": day,
                "epoch": epoch + 1,
                "epochs": ATTENTION_RECIPE["epochs"],
                "loss": total_loss / event_count,
            },
            flush=True,
        )

    checkpoint = output / "attention.pt"
    torch.save(attention.state_dict(), checkpoint)
    val, val_features = _load_validation(root, validation, h)
    val_events = np.asarray(val["events"]).reshape(-1, 10)[:, 0]
    if len(val_events) * 10 != len(val["y"]):
        raise AssertionError("Validation event IDs are not aligned with labels")
    if not np.all(np.asarray(val["events"]).reshape(-1, 10) == val_events[:, None]):
        raise AssertionError("Validation event groups are malformed")
    with np.load(neural_root / f"val{day}.npz") as neural_validation:
        if not np.array_equal(val_events, neural_validation["indices"]):
            raise AssertionError("Feature and history validation event IDs differ")
        if not np.array_equal(val["y"], neural_validation["y"]):
            raise AssertionError("Feature and history validation labels differ")
        if not np.allclose(
            val["baseline"], neural_validation["baseline"], rtol=0, atol=1e-7
        ):
            raise AssertionError("Feature and history validation baselines differ")
        if not np.allclose(
            val["weights"], neural_validation["weights"], rtol=0, atol=1e-7
        ):
            raise AssertionError("Feature and history validation weights differ")
    if not np.array_equal(
        event_data["target"][val_events], _target_from_labels(val["y"])
    ):
        raise AssertionError("Saved V53 validation labels and event targets differ")
    if not (event_data["timestamp"][val_events] >= int(validation["cutoff"])).all():
        raise AssertionError("Validation events do not begin at the declared cutoff")
    raw = _prediction(attention, val_features, event_data, val_events, device)
    parent = _prediction(
        model, val_features, event_data, val_events, device
    )
    parent_reloaded = ChoiceNet(
        model.mean.detach().cpu(), model.scale.detach().cpu(), PARENT_RECIPE
    ).to(device)
    parent_reloaded.load_state_dict(
        torch.load(output.parent / "parent" / "parent.pt", map_location=device, weights_only=True)
    )
    if not np.array_equal(
        _prediction(parent_reloaded, val_features, event_data, val_events, device),
        parent,
    ):
        raise AssertionError("Saved parent checkpoint replay changed")
    if not np.isfinite(raw).all() or not np.isfinite(parent).all():
        raise AssertionError("Non-finite validation predictions")
    pred = h.rank(
        (1.0 - ATTENTION_RECIPE["blend_weight"]) * h.rank(val["baseline"])
        + ATTENTION_RECIPE["blend_weight"] * h.rank(raw)
    )
    parent_blend = h.rank(
        (1.0 - ATTENTION_RECIPE["blend_weight"]) * h.rank(val["baseline"])
        + ATTENTION_RECIPE["blend_weight"] * h.rank(parent)
    )
    report = h.metrics(val["y"], pred, val["baseline"], val["weights"])
    vs_parent = h.metrics(val["y"], pred, parent_blend, val["weights"])

    restored = HistoryChoice(
        ChoiceNet(model.mean.detach().cpu(), model.scale.detach().cpu(), PARENT_RECIPE),
        item_vectors,
        history_vectors,
        scale=ATTENTION_RECIPE["maximum_logit_delta"],
    ).to(device)
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    replay = _prediction(restored, val_features, event_data, val_events, device)
    if not np.array_equal(replay, raw):
        raise AssertionError("History-attention checkpoint replay changed")
    report.update(
        day=day,
        raw_auc=float(roc_auc_score(val["y"], raw)),
        parent_raw_auc=float(roc_auc_score(val["y"], parent)),
        vs_parent_blend=vs_parent,
        saved_checkpoint_replay="PASS",
        model_sha256=_digest(checkpoint),
        rows=len(raw),
        events=len(val_events),
        csv_ready=False,
        official_auc=None,
        target_official_auc=0.82,
        official_target_achieved=None,
        note="Reused development cohort; local validation metrics do not predict official leaderboard AUC.",
    )
    np.savez_compressed(
        output / "predictions.npz",
        raw=raw,
        parent_raw=parent,
        pred=pred,
        parent_blend=parent_blend,
        **val,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def run(root, neural_root, output, weighting_path, device="cuda"):
    root = Path(root).resolve()
    neural_root = Path(neural_root).resolve()
    output = Path(output).resolve()
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this experiment")
    torch.set_num_threads(4)

    h = _load_module("attention_feature_helpers", root / "runner.py")
    weighting = _load_module("attention_history_weights", weighting_path)
    manifest = h.load_bundle(root, verify=True)
    neural_manifest = json.loads((neural_root / "neural_manifest.json").read_text())
    for name, metadata in neural_manifest["files"].items():
        path = (neural_root / name).resolve()
        if not path.is_relative_to(neural_root):
            raise ValueError(f"Neural bundle path escapes its root: {name}")
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Neural bundle size check failed: {name}")
        if _digest(path) != metadata["sha256"]:
            raise ValueError(f"Neural bundle checksum failed: {name}")
    with np.load(neural_root / "events.npz") as archive:
        event_data = {key: archive[key].copy() for key in archive.files}
    reports = []
    for day in (6, 5):
        day_output = output / f"day{day}"
        day_output.mkdir(parents=True, exist_ok=False)
        parent_dir = day_output / "parent"
        model, train_features, targets, event_weights, parent_meta = _train_parent(
            root, neural_root, day, manifest, weighting, h, device, parent_dir
        )
        specs = [manifest["train"][str(d)] for d in range(day - 3, day)]
        labels = np.concatenate(
            [np.load(root / spec["labels"]) for spec in specs]
        ).astype(np.float32)
        fitting_ids, targets = _aligned_events(
            event_data, neural_manifest, day, labels
        )
        if len(fitting_ids) != len(targets) or len(event_weights) != len(targets):
            raise AssertionError("Fitting features, labels, and event rows are misaligned")
        validation = manifest["validation"][str(day)]
        report = _train_attention(
            root,
            day,
            model,
            train_features,
            event_data,
            fitting_ids,
            targets,
            event_weights,
            neural_root,
            validation,
            h,
            device,
            day_output / "attention",
        )
        report["parent"] = parent_meta
        reports.append(report)
        del model
        if report["adjusted_gain"] <= 0 or report["adjusted_gain_interval"][0] <= 0:
            break
        if report["vs_parent_blend"]["adjusted_gain"] <= 0:
            break

    summary = {
        "experiment": "ctr-history-attention-001",
        "reports": reports,
        "eligible_for_review": bool(
            len(reports) == 2
            and all(r["adjusted_gain"] > 0 for r in reports)
            and all(r["adjusted_gain_interval"][0] > 0 for r in reports)
            and all(r["vs_parent_blend"]["adjusted_gain"] > 0 for r in reports)
        ),
        "target_metric": "official leaderboard ROC AUC",
        "target_official_auc": 0.82,
        "official_auc": None,
        "target_achieved": None,
        "promotion_blocked": True,
        "note": "Official performance can only be measured by a valid submission; these reused validation cohorts are exploratory.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
