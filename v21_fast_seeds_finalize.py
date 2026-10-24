"""Final fit of the confirmed multi-seed fast DIN recipe -> submission_v21.csv.

Requires both-day gain over the single-seed V21. Never touches submission.csv.
"""
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
    NI,
    SEED,
    START,
    append_log,
    ehist,
    etimes,
    evaluation,
    hist,
    htimes,
    pack_events,
    predict_probs,
    train,
)
from v21_fast_din import FastDIN
from v21_fast_seeds import FAST1, rank, run_epoch

torch.set_num_threads(14)
BATCH = 768


def main():
    t0 = time.time()
    sel = json.loads((ROOT / "v21_fastseeds_day6_report.json").read_text())
    seeds = sel["seeds"]
    # Weight stays at the previously confirmed 0.20; only seed-averaging changes.
    wf = 0.2
    scores = {}
    for day in (6, 5):
        zs = [np.load(ROOT / f"v21_fastseeds_day{day}_s{s}_ep1_raw.npz") for s in seeds]
        y, base = zs[0]["y"], zs[0]["base"]
        mean3 = np.mean([rank(z["raw"]) for z in zs], 0)
        single = rank(np.load(ROOT / f"v21_fastdin_day{day}_ep1_raw.npz")["raw"])
        from sklearn.metrics import roc_auc_score

        scores[day] = {
            "single": float(roc_auc_score(y, (1 - wf) * rank(base) + wf * single)),
            "mean": float(roc_auc_score(y, (1 - wf) * rank(base) + wf * mean3)),
        }
        assert scores[day]["mean"] > scores[day]["single"], f"no gain on day {day}: {scores[day]}"
    best = {"composite": "fast1_rankmean", "w_fast": wf, "w_old": 0.0, "auc": scores[6]["mean"]}
    conf = {"auc": scores[5]["mean"]}
    print(json.dumps(scores), flush=True)

    rows = np.asarray(train[(train[:, 2] >= START + 4 * DAY) & (train[:, 2] < START + 7 * DAY)])
    assert int(evaluation[:, 2].min()) >= START + 7 * DAY
    data = pack_events(rows, hist, htimes)
    idx = np.where(data["has_click"])[0]
    edata = pack_events(np.asarray(evaluation), ehist, etimes)
    base_state = torch.load(ROOT / "v21_din_frozen_als_day6.pt", map_location="cpu", weights_only=True)
    print(f"final events={data['n']} clicks={len(idx)} seeds={seeds} w_fast={wf}", flush=True)

    ranks = []
    for s in seeds:
        seed = SEED + 107 + 1000 * s
        torch.manual_seed(seed)
        model = FastDIN(NI, dim=64, rank=16)
        model.load_state_dict(base_state, strict=False)
        model.item.weight.requires_grad_(False)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-5)
        loss = run_epoch(model, opt, data, idx, np.random.default_rng(seed), BATCH)
        din = predict_probs(model, edata).astype(np.float64)
        assert len(din) == 529390 and np.isfinite(din).all()
        np.save(ROOT / f"v21_fastseeds_final_s{s}_eval.npy", din.astype(np.float32))
        torch.save(model.state_dict(), ROOT / f"v21_fastseeds_final_s{s}.pt")
        ranks.append(rank(din))
        print(json.dumps({"seed": s, "loss": loss}), flush=True)
    fast = np.mean(ranks, 0)

    v20 = np.empty(529390, dtype=np.float64)
    with (ROOT / "submission_v20.csv").open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            v20[i] = float(row["predicted_probability"])
    pred = np.clip((1 - wf) * rank(v20) + wf * rank(fast), 0, 1)

    prev = Path("submission_v21.csv")
    if prev.exists():
        backup = ROOT / "submission_v21_singleseed.csv"
        if not backup.exists():
            backup.write_bytes(prev.read_bytes())
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
        "recipe": best,
        "seeds": seeds,
        "local_day6": best["auc"],
        "local_day5": conf["auc"],
        "single_seed_same_weight": {str(d): scores[d]["single"] for d in scores},
        "prior_single_seed": FAST1,
        "runtime_s": time.time() - t0,
        "submission_csv_untouched": True,
    }
    (ROOT / "v21_fastseeds_manifest.json").write_text(json.dumps(meta, indent=2))
    append_log(
        "V21-fastseeds-final",
        f"{len(seeds)}-seed fast DIN rank-mean, w={wf} -> submission_v21.csv",
        FAST1[6],
        best["auc"],
        meta["runtime_s"],
        True,
        f"day5={conf['auc']:.6f}",
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
