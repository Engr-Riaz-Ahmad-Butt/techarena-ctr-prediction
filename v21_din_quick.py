"""Quick DIN: frozen SVD emb + logits AUC; then short unfreeze."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from v21_din import (
    DAY,
    DIN,
    NI,
    ROOT,
    SEED,
    START,
    batch_loss,
    event_batches,
    fit_rows,
    hist,
    htimes,
    load_v20,
    pack_events,
    predict_logits,
    predict_probs,
)
from v21_experiments import item_embeddings

torch.manual_seed(SEED)
np.random.seed(SEED)

val, y, events, base = load_v20(6)
fit = fit_rows(6)
td = pack_events(fit, hist, htimes)
vd = pack_events(val, hist, htimes)
emb, _ = item_embeddings(START + 4 * DAY, dim=64)

model = DIN(NI, dim=64)
with torch.no_grad():
    model.item.weight[:-1].copy_(torch.from_numpy(emb))
    model.item.weight[-1].zero_()
model.item.weight.requires_grad_(False)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
rng = np.random.default_rng(SEED)
idx = np.where(td["has_click"])[0]
best = {"auc": -1.0}
hist_rows = []
for epoch in range(1, 9):
    model.train()
    model.item.weight.requires_grad_(False)
    tot = 0.0
    n = 0
    for batch in event_batches(td, idx, bs=256, shuffle=True, rng=rng):
        batch["mask"] = batch["mask"].bool()
        batch["has_click"] = batch["has_click"].bool()
        opt.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        opt.step()
        tot += float(loss.detach()) * batch["hist"].size(0)
        n += batch["hist"].size(0)
    logits = predict_logits(model, vd)
    probs = predict_probs(model, vd)
    auc_l = float(roc_auc_score(y, logits))
    auc_p = float(roc_auc_score(y, probs))
    row = {"epoch": epoch, "loss": tot / max(n, 1), "auc_logits": auc_l, "auc_probs": auc_p}
    hist_rows.append(row)
    print(json.dumps(row), flush=True)
    if auc_l > best["auc"]:
        best = {"auc": auc_l, "epoch": epoch, "raw": logits, "kind": "logits"}
        torch.save(model.state_dict(), ROOT / "v21_din_frozen_best.pt")

np.savez_compressed(
    ROOT / "v21_din_frozen_day6_raw.npz", y=y, events=events, raw=best["raw"], base=base
)
report = {
    "best_auc_logits": best["auc"],
    "best_epoch": best["epoch"],
    "v20": float(roc_auc_score(y, base)),
    "history": hist_rows,
}
(ROOT / "v21_din_frozen_report.json").write_text(json.dumps(report, indent=2))
print("BEST", report["best_auc_logits"], flush=True)
