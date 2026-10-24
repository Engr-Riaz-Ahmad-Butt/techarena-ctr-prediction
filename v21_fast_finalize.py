"""One-epoch fast DIN on the pre-evaluation window, then write submission_v21.csv.

Weight 0.20 was frozen on day 6 and confirmed on day 5. Does not touch submission.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from config import ARTIFACT_DIR as ROOT
from config import EVALUATION_CSV
from v21_din import (
    DAY,
    NI,
    SEED,
    START,
    append_log,
    batch_loss,
    ehist,
    etimes,
    evaluation,
    event_batches,
    hist,
    htimes,
    pack_events,
    predict_probs,
    train,
)
from v21_fast_din import BEST_BLEND, FastDIN

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(14)

WEIGHT = 0.2
BATCH = 768


def main():
    rows = np.asarray(train[(train[:, 2] >= START + 4 * DAY) & (train[:, 2] < START + 7 * DAY)])
    assert int(evaluation[:, 2].min()) >= START + 7 * DAY
    data = pack_events(rows, hist, htimes)
    model = FastDIN(NI, dim=64, rank=16)
    state = torch.load(ROOT / "v21_din_frozen_als_day6.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.item.weight.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-5)
    rng = np.random.default_rng(SEED + 107)
    idx = np.where(data["has_click"])[0]
    print(f"final fast DIN events={data['n']} clicks={len(idx)}", flush=True)
    model.train()
    tot = n = 0.0
    for batch in event_batches(data, idx, bs=BATCH, shuffle=True, rng=rng):
        batch["mask"] = batch["mask"].bool()
        batch["has_click"] = batch["has_click"].bool()
        opt.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        opt.step()
        tot += float(loss.detach()) * batch["hist"].size(0)
        n += batch["hist"].size(0)
    print(f"final loss={tot / max(n, 1):.4f}", flush=True)
    torch.save(model.state_dict(), ROOT / "v21_fastdin_final.pt")

    edata = pack_events(np.asarray(evaluation), ehist, etimes)
    din = predict_probs(model, edata).astype(np.float64)
    assert len(din) == 529390 and np.isfinite(din).all()
    np.save(ROOT / "v21_fastdin_eval.npy", din.astype(np.float32))

    v20 = np.empty(529390, dtype=np.float64)
    with (ROOT / "submission_v20.csv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            v20[i] = float(row["predicted_probability"])
    pred = (1 - WEIGHT) * rankdata(v20, method="average") / (len(v20) + 1)
    pred += WEIGHT * rankdata(din, method="average") / (len(din) + 1)
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
    eval_ids = [r["rowid"] for r in csv.DictReader(EVALUATION_CSV.open(newline="", encoding="utf-8"))]
    sub_ids = [r["rowid"] for r in csv.DictReader(out.open(newline="", encoding="utf-8"))]
    assert eval_ids == sub_ids
    meta = {
        "rows": n,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "weight_v20": 1 - WEIGHT,
        "weight_fastdin": WEIGHT,
        "local_day6_blend": 0.8239436671631495,
        "local_day5_blend": 0.8165425964775535,
        "prior_day6_blend": BEST_BLEND[6],
        "prior_day5_blend": BEST_BLEND[5],
        "submission_csv_untouched": True,
    }
    (ROOT / "v21_fastdin_manifest.json").write_text(json.dumps(meta, indent=2))
    append_log(
        "V21-fastDIN",
        "1-epoch vectorized DIN + rank-16 adapter, 20% rank blend",
        BEST_BLEND[6],
        0.8239436671631495,
        0,
        True,
        "day5 0.816543; submission_v21.csv only",
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
