"""Final DIN fit + rank-blend with V20 -> submission_v21.csv (does not touch submission.csv)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from config import ARTIFACT_DIR as ROOT
from config import EVALUATION_CSV
from v21_din import (
    DAY,
    DIN,
    NI,
    SEED,
    START,
    append_log,
    batch_loss,
    event_batches,
    evaluation,
    ehist,
    etimes,
    hist,
    htimes,
    init_embeddings,
    pack_events,
    predict_probs,
    train,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(6)

WEIGHTS = json.loads((ROOT / "v21_final_blend_weights.json").read_text())
EPOCHS = int(json.loads((ROOT / "v21_din_frozen_als_day6_report.json").read_text())["best_epoch"])
DIM = 64
LR = 3e-3


def main():
    t0 = time.time()
    report = json.loads((ROOT / "v21_final_blend_report.json").read_text())
    assert report["kept"], "Blend gate failed; refusing to write submission_v21.csv"

    # Final fit on late training days (same spirit as stack finalize)
    rows = np.asarray(train[train[:, 2] >= START + 3 * DAY])
    data = pack_events(rows, hist, htimes)
    model = DIN(NI, dim=DIM)
    init_embeddings(model, "als", als_cut=7, dim=DIM)  # pre-evaluation ALS checkpoint
    model.item.weight.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    idx = np.where(data["has_click"])[0]
    print(f"Final DIN: events={data['n']} clicks={len(idx)} epochs={EPOCHS}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.item.weight.requires_grad_(False)
        tot = n = 0.0
        for batch in event_batches(data, idx, bs=256, shuffle=True, rng=rng):
            batch["mask"] = batch["mask"].bool()
            batch["has_click"] = batch["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(model, batch)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * batch["hist"].size(0)
            n += batch["hist"].size(0)
        print(f"final epoch {epoch}/{EPOCHS} loss={tot/max(n,1):.4f}", flush=True)
        torch.save(model.state_dict(), ROOT / f"v21_din_final_ep{epoch}.pt")

    torch.save(model.state_dict(), ROOT / "v21_din_final.pt")
    edata = pack_events(np.asarray(evaluation), ehist, etimes)
    din = predict_probs(model, edata).astype(np.float64)
    assert len(din) == 529390 and np.isfinite(din).all()
    np.save(ROOT / "v21_din_eval.npy", din.astype(np.float32))

    v20 = np.empty(529390, dtype=np.float64)
    with (ROOT / "submission_v20.csv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            v20[i] = float(row["predicted_probability"])

    parts = {"v20": v20, "din_als": din}
    ranks = {k: rankdata(v, method="average") / (len(v) + 1) for k, v in parts.items()}
    pred = np.zeros(529390, dtype=np.float64)
    for k, w in WEIGHTS.items():
        pred += w * ranks[k]
    pred = np.clip(pred, 0, 1)

    out = Path("submission_v21.csv")
    with EVALUATION_CSV.open(newline="", encoding="utf-8") as src, out.open(
        "w", newline="", encoding="utf-8"
    ) as dest:
        w = csv.writer(dest)
        w.writerow(["rowid", "predicted_probability"])
        n = 0
        for row, p in zip(csv.DictReader(src), pred, strict=True):
            assert math.isfinite(p) and 0.0 <= p <= 1.0
            w.writerow([row["rowid"], format(float(p), ".10g")])
            n += 1
    assert n == 529390
    (ROOT / "submission_v21.csv").write_bytes(out.read_bytes())

    # verify order vs evaluation
    eval_ids = [r["rowid"] for r in csv.DictReader(EVALUATION_CSV.open(newline="", encoding="utf-8"))]
    sub_ids = [r["rowid"] for r in csv.DictReader(out.open(newline="", encoding="utf-8"))]
    assert eval_ids == sub_ids
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "file": str(out.resolve()),
        "rows": n,
        "sha256": digest,
        "weights": WEIGHTS,
        "local_day6_blend": report["days"]["6"]["blend_auc"],
        "local_day5_blend": report["days"]["5"]["blend_auc"],
        "runtime_s": time.time() - t0,
        "submission_csv_untouched": True,
    }
    (ROOT / "v21_submission_manifest.json").write_text(json.dumps(meta, indent=2))
    append_log(
        "V21-final",
        "Wrote submission_v21.csv (V20+DIN rank blend)",
        report["days"]["6"]["v20_auc"],
        report["days"]["6"]["blend_auc"],
        meta["runtime_s"],
        True,
        f"day5={report['days']['5']['blend_auc']:.6f} w={WEIGHTS}",
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
