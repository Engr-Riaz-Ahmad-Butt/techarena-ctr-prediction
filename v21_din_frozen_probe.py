"""Frozen SVD item embeddings + trainable DIN attention/MLP only."""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from v21_din import (
    DAY,
    DIN,
    NI,
    SEED,
    START,
    ROOT,
    batch_loss,
    event_batches,
    fit_rows,
    hist,
    htimes,
    load_v20,
    pack_events,
    predict_probs,
)
from v21_experiments import item_embeddings

torch.manual_seed(SEED)
np.random.seed(SEED)

day = 6
val, y, events, base = load_v20(day)
fit = fit_rows(day)
train_data = pack_events(fit, hist, htimes)
val_data = pack_events(val, hist, htimes)
emb, _ = item_embeddings(START + (day - 2) * DAY, dim=64)

model = DIN(NI, dim=64)
with torch.no_grad():
    model.item.weight[:-1].copy_(torch.from_numpy(emb))
    model.item.weight[-1].zero_()
model.item.weight.requires_grad_(False)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-5)
rng = np.random.default_rng(SEED)
idx = np.where(train_data["has_click"])[0]
print(f"frozen-emb probe click events={len(idx)}", flush=True)

best = -1.0
history = []
for epoch in range(1, 11):
    model.train()
    # keep emb frozen
    model.item.weight.requires_grad_(False)
    total = 0.0
    n = 0
    for batch in event_batches(train_data, idx, bs=256, shuffle=True, rng=rng):
        batch["mask"] = batch["mask"].bool()
        batch["has_click"] = batch["has_click"].bool()
        opt.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * batch["hist"].size(0)
        n += batch["hist"].size(0)
    raw = predict_probs(model, val_data)
    auc = float(roc_auc_score(y, raw))
    acc = float(
        (raw.reshape(-1, 10).argmax(1) == val_data["target"])[val_data["has_click"]].mean()
    )
    row = {"epoch": epoch, "loss": total / max(n, 1), "auc": auc, "click_acc": acc}
    history.append(row)
    print(json.dumps(row), flush=True)
    best = max(best, auc)

(ROOT / "v21_din_frozen_probe.json").write_text(
    json.dumps({"best_auc": best, "history": history, "v20": float(roc_auc_score(y, base))}, indent=2)
)
print("BEST", best, flush=True)
