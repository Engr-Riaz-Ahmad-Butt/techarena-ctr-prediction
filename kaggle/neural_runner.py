"""Jointly finetune all warm user/item/history embeddings on preceding events.

Fixed six epochs, fixed residual blend, and unchanged day5 confirmation after
a day6 gate. Validation labels never enter optimization or epoch selection.
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


class JointModel(nn.Module):
    def __init__(self, factors, context):
        super().__init__()
        self.dim = factors["V"].shape[1]
        for name in ["U", "V", "H"]:
            values = torch.from_numpy(factors[name])
            setattr(self, name, nn.Embedding.from_pretrained(values.clone(), freeze=False))
        for name in ["bu", "bi"]:
            values = torch.from_numpy(factors[name][:, None])
            setattr(self, name, nn.Embedding.from_pretrained(values.clone(), freeze=False))
        self.bucket_matrix = nn.Parameter(context["bucket"].clone())
        self.kind_matrix = nn.Parameter(context["kind"].clone())
        self.recent_matrix = nn.Parameter(context["recent"].clone())

    def forward(self, users, items, hist, gaps, bucket, kind):
        mask = hist >= 0
        history_vectors = self.H(hist.clamp(min=0)) * mask[:, :, None]
        length = mask.sum(1).clamp(min=1).float()
        z = self.U(users) + history_vectors.sum(1) / length.sqrt()[:, None]
        weight = torch.exp(-np.log(2)*gaps/2) * mask
        recent = (history_vectors * weight[:, :, None]).sum(1) / weight.sum(1).clamp(min=1).sqrt()[:, None]
        v = self.V(items)
        base = (-1.8 + self.bu(users) + self.bi(items).squeeze(-1) + (z[:, None]*v).sum(-1)).clamp(-30, 30)
        matrix = self.bucket_matrix[bucket] + self.kind_matrix[kind]
        delta = torch.bmm(z[:, None], matrix).squeeze(1) + recent @ self.recent_matrix
        logits = base + (delta[:, None]*v).sum(-1)
        return torch.cat([logits, torch.zeros(len(users), 1, device=logits.device)], 1)


def batch(data, ix, device):
    return tuple(torch.as_tensor(data[k][ix], device=device,
                 dtype=torch.float32 if k == "gaps" else torch.long)
                 for k in ["users", "items", "hist", "gaps", "bucket", "kind"])


@torch.no_grad()
def predict(model, data, indices, device):
    model.eval()
    return np.concatenate([model(*batch(data, indices[off:off+2048], device)).softmax(1)[:, :10].cpu().numpy()
                           for off in range(0, len(indices), 2048)]).ravel()


def fit(root, output, day, recipe, helpers, device):
    torch.manual_seed(recipe["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(recipe["seed"])
    torch.set_num_threads(4)
    rng = np.random.default_rng(recipe["seed"])
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = json.loads((root / "neural_manifest.json").read_text())
    data = dict(np.load(root / "events.npz"))
    cut = m["start"] + day*86400
    fitting = np.flatnonzero((data["timestamp"] >= cut-recipe["fit_days"]*86400) & (data["timestamp"] < cut))
    val = dict(np.load(root / f"val{day}.npz"))
    vi = val["indices"]
    assert data["timestamp"][fitting].max() < data["timestamp"][vi].min()
    assert not np.intersect1d(fitting, vi).size
    with np.load(root / f"factors{day}.npz") as z:
        factors = {k:z[k].astype("float32") for k in ["U", "V", "H", "bu", "bi"]}
    context = torch.load(root / f"context{day}.pt", map_location="cpu", weights_only=True)
    model = JointModel(factors, context).to(device)
    initial = predict(model, data, vi, device)
    expected = np.load(root / f"initial{day}.npy")
    replay_error = float(np.max(np.abs(initial-expected)))
    assert replay_error < 3e-6, ("Initial V36 replay mismatch", replay_error)
    anchors = {name:p.detach().clone() for name,p in model.named_parameters()}
    # Penalize drift from the pretrained solution, rather than shrinking its
    # useful item/history vectors toward zero. Embeddings remain trainable.
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    started = time.monotonic()
    (output / "recipe.json").write_text(json.dumps(dict(recipe=recipe,day=day,
        input_manifest_sha256=digest(root/"neural_manifest.json"), torch=torch.__version__,
        first_fit_timestamp=int(data["timestamp"][fitting].min()),last_fit_timestamp=int(data["timestamp"][fitting].max()),
        validation_min_timestamp=int(data["timestamp"][vi].min()),initial_replay_error=replay_error), indent=2))
    for epoch in range(recipe["epochs"]):
        model.train()
        order = rng.permutation(fitting)
        loss_total = 0.
        for off in range(0, len(order), recipe["batch_size"]):
            ix = order[off:off+recipe["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch(data, ix, device))
            y = torch.as_tensor(data["target"][ix], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(logits, y)
            penalty = sum((p-anchors[name]).square().mean() for name,p in model.named_parameters())
            (loss + recipe["anchor_penalty"]*penalty).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_total += float(loss.detach())*len(ix)
        progress = dict(day=day,epoch=epoch+1,epochs=recipe["epochs"],loss=loss_total/len(order),elapsed_s=round(time.monotonic()-started,1))
        print(progress, flush=True)
        (output / "progress.json").write_text(json.dumps(progress))
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    raw = predict(model, data, vi, device)
    # Replace a fixed fraction of the original neural component in V53.
    pred = helpers.rank(helpers.rank(val["baseline"]) + recipe["rank_residual_weight"]*(helpers.rank(raw)-helpers.rank(initial)))
    metrics = helpers.metrics(val["y"],pred,val["baseline"],val["weights"])
    metrics.update(day=day,raw_auc=float(helpers.roc_auc_score(val["y"],raw)),initial_raw_auc=float(helpers.roc_auc_score(val["y"],initial)),
                   initial_replay_error=replay_error,model_sha256=digest(checkpoint),csv_ready=False,official_auc=None)
    restored = JointModel(factors, context).to(device)
    restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    reproduced = predict(restored,data,vi,device)
    assert np.array_equal(reproduced,raw), "Saved neural model replay changed"
    metrics["saved_checkpoint_replay"] = "PASS"
    np.savez_compressed(output/"predictions.npz",raw=raw,initial=initial,pred=pred,**val)
    (output/"report.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics


def run(root, output, device="cuda"):
    root,output = Path(root),Path(output)
    assert device == "cpu" or torch.cuda.is_available(), "Enable Kaggle GPU"
    manifest = json.loads((root/"neural_manifest.json").read_text())
    for name,meta in manifest["files"].items():
        path=(root/name).resolve()
        assert path.is_relative_to(root.resolve())
        assert path.stat().st_size==meta["bytes"] and digest(path)==meta["sha256"],name
    output.mkdir(parents=True,exist_ok=True)
    # Never silently reuse trained neural weights for a different experiment.
    assert not (output/"day6"/"model.pt").exists(), "Choose a fresh output directory"
    spec=importlib.util.spec_from_file_location("ctr_metric_helpers",root/"runner.py")
    helpers=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    reports=[fit(root,output/"day6",6,manifest["recipe"],helpers,device)]
    if reports[0]["positive_gate"]:
        reports.append(fit(root,output/"day5",5,manifest["recipe"],helpers,device))
    summary=dict(reports=reports,eligible_for_review=len(reports)==2 and all(r["positive_gate"] for r in reports),
                 target=.83,target_achieved=False,csv_ready=False,official_auc=None,
                 note="Reused local dates, fixed recipe, no official-score forecast or competition CSV.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary
