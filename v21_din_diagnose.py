"""Diagnose packing + mid-size DIN train (click events only)."""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from v21_din import (
    DIN,
    NI,
    PAD,
    START,
    DAY,
    ROOT,
    fit_rows,
    load_v20,
    pack_events,
    hist,
    htimes,
    init_embeddings,
    predict_probs,
    batch_loss,
    SEED,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

val, y, events, base = load_v20(6)
fit = fit_rows(6)
print("packing...", flush=True)
train_data = pack_events(fit, hist, htimes)
val_data = pack_events(val, hist, htimes)

# Verify packing vs raw history for 200 random events
rng = np.random.default_rng(0)
for i in rng.choice(train_data["n"], size=200, replace=False):
    L = int(train_data["mask"][i].sum())
    eid = int(fit[i * 10, 6])
    h = np.asarray(hist[eid])
    raw = h[h >= 0]
    assert L == len(raw), (L, len(raw), i)
    if L:
        assert np.array_equal(train_data["hist"][i, -L:], raw), i
        assert train_data["hist"][i, -1] == raw[-1]
print("Packing verification PASSED on 200 events", flush=True)

# Label check
assert np.allclose(train_data["labels"].ravel(), fit[:, 5])
assert np.allclose(val_data["labels"].ravel(), val[:, 5])
print("Label alignment PASSED", flush=True)

# Train on 8k click events only, high LR
click = np.where(train_data["has_click"])[0]
rng.shuffle(click)
click = click[:8000]
model = DIN(NI, dim=64)
init_embeddings(model, "als", als_cut=4, dim=64)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)

def subset(data, idx):
    out = {k: (v[idx] if isinstance(v, np.ndarray) and k != "n" else v) for k, v in data.items()}
    out["n"] = len(idx)
    return out

sub = subset(train_data, click)
for epoch in range(1, 9):
    model.train()
    # one big-ish shuffled pass in mini-batches
    order = np.arange(sub["n"])
    rng.shuffle(order)
    total = 0.0
    for s in range(0, len(order), 256):
        sl = order[s : s + 256]
        batch = {
            "hist": torch.from_numpy(sub["hist"][sl]),
            "mask": torch.from_numpy(sub["mask"][sl]),
            "gaps": torch.from_numpy(sub["gaps"][sl]),
            "cands": torch.from_numpy(sub["cands"][sl]),
            "bucket": torch.from_numpy(sub["bucket"][sl]),
            "day_type": torch.from_numpy(sub["day_type"][sl]),
            "target": torch.from_numpy(sub["target"][sl]),
            "has_click": torch.from_numpy(sub["has_click"][sl]),
            "labels": torch.from_numpy(sub["labels"][sl]),
        }
        opt.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(sl)
    raw = predict_probs(model, val_data)
    auc = float(roc_auc_score(y, raw))
    print(f"epoch {epoch} loss={total/len(order):.4f} val_auc={auc:.6f}", flush=True)

(ROOT / "v21_din_diagnose.json").write_text(
    json.dumps({"note": "see stdout"}, indent=2)
)
