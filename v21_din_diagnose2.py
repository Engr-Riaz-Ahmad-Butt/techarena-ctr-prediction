import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from v21_din import (
    DIN,
    NI,
    SEED,
    batch_loss,
    fit_rows,
    hist,
    htimes,
    init_embeddings,
    load_v20,
    pack_events,
    predict_probs,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
val, y, events, base = load_v20(6)
fit = fit_rows(6)
train_data = pack_events(fit, hist, htimes)
val_data = pack_events(val, hist, htimes)
rng = np.random.default_rng(0)
click = np.where(train_data["has_click"])[0]
rng.shuffle(click)
click = click[:8000]
model = DIN(NI, dim=64)
init_embeddings(model, "random", 4, 64)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(6):
    model.train()
    order = np.arange(len(click))
    rng.shuffle(order)
    tot = 0.0
    for s in range(0, len(order), 256):
        sl = click[order[s : s + 256]]
        batch = {
            k: torch.from_numpy(train_data[k][sl])
            for k in [
                "hist",
                "mask",
                "gaps",
                "cands",
                "bucket",
                "day_type",
                "target",
                "has_click",
                "labels",
            ]
        }
        opt.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        opt.step()
        tot += float(loss) * len(sl)
    sub = {
        k: (train_data[k][click] if isinstance(train_data[k], np.ndarray) and k != "n" else train_data[k])
        for k in train_data
    }
    sub["n"] = len(click)
    raw_tr = predict_probs(model, sub)
    y_tr = train_data["labels"][click].ravel()
    raw_va = predict_probs(model, val_data)
    tr_acc = (raw_tr.reshape(-1, 10).argmax(1) == train_data["target"][click]).mean()
    va_acc = (raw_va.reshape(-1, 10).argmax(1) == val_data["target"])[val_data["has_click"]].mean()
    print(
        f"ep{epoch+1} loss={tot/len(click):.3f} "
        f"train_auc={roc_auc_score(y_tr, raw_tr):.4f} val_auc={roc_auc_score(y, raw_va):.4f} "
        f"train_acc={tr_acc:.3f} val_click_acc={va_acc:.3f}",
        flush=True,
    )
