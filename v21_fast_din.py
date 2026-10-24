"""Vectorized DIN plus a zero-init item adapter.

Warm-starts from the frozen-ALS checkpoint, keeps base embeddings frozen,
and rank-blends against V20 on the same first-event cohort. Does not write
submission.csv.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from v21_din import (
    DAY,
    DIN,
    NI,
    ROOT,
    SEED,
    START,
    V20,
    append_log,
    batch_loss,
    event_batches,
    fit_rows,
    hist,
    htimes,
    init_embeddings,
    load_v20,
    pack_events,
    predict_probs,
)

BEST_BLEND = {6: 0.8237285378313302, 5: 0.816311198655789}


class FastDIN(DIN):
    """Same scorer as DIN, with all 10 candidates in one matmul, plus LoRA."""

    def __init__(self, n_items: int, dim: int = 64, rank: int = 16):
        super().__init__(n_items, dim=dim)
        self.lora_a = nn.Embedding(n_items + 1, rank, padding_idx=n_items)
        self.lora_b = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        with torch.no_grad():
            self.lora_a.weight[n_items].zero_()
            self.lora_b.weight.zero_()

    def _embed(self, ids):
        ids = ids.clamp(max=self.pad)
        return self.item(ids) + self.lora_b(self.lora_a(ids))

    def forward(self, hist, mask, gaps, cands, bucket, day_type):
        he = self._embed(hist) + self.gap_emb(gaps.unsqueeze(-1))
        ce = self._embed(cands)
        he_e = he[:, None, :, :]
        ce_e = ce[:, :, None, :]
        cat = torch.cat(
            [
                he_e.expand(-1, 10, -1, -1),
                ce_e.expand(-1, -1, 20, -1),
                he_e * ce_e,
                he_e - ce_e,
            ],
            dim=-1,
        )
        score = self.attn(cat).squeeze(-1)
        score = score.masked_fill(~mask[:, None, :], -1e9)
        all_pad = ~mask.any(dim=1)
        score = score.masked_fill(all_pad[:, None, None], 0.0)
        attn = torch.softmax(score, dim=-1)
        attn = attn.masked_fill(all_pad[:, None, None], 0.0)
        ctx = (attn.unsqueeze(-1) * he[:, None, :, :]).sum(2)
        b = self.bucket(bucket)
        d = self.daytype(day_type)
        empty = all_pad.float()[:, None, None].expand(-1, 10, 1)
        feat = torch.cat(
            [
                ctx,
                ce,
                b[:, None, :].expand(-1, 10, -1),
                d[:, None, :].expand(-1, 10, -1),
                empty,
            ],
            dim=-1,
        )
        return self.mlp(feat).squeeze(-1)


def _batch(n=4, dim_seed=0):
    g = torch.Generator().manual_seed(dim_seed)
    hist = torch.randint(0, NI, (n, 20), generator=g)
    mask = torch.rand(n, 20, generator=g) > 0.3
    mask[0].zero_()
    gaps = torch.rand(n, 20, generator=g)
    cands = torch.randint(0, NI, (n, 10), generator=g)
    bucket = torch.randint(0, 4, (n,), generator=g)
    day = torch.randint(0, 2, (n,), generator=g)
    return hist, mask, gaps, cands, bucket, day


def check_parity():
    torch.manual_seed(0)
    base = DIN(NI, dim=64)
    fast = FastDIN(NI, dim=64, rank=16)
    fast.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    fast.eval()
    batch = _batch(8)
    with torch.no_grad():
        a = base(*batch)
        b = fast(*batch)
    err = (a - b).abs().max().item()
    print(f"max_logit_diff={err:.3e}", flush=True)
    if err > 1e-4:
        raise SystemExit("vectorized DIN does not match the loop")
    # speed on a realistic batch
    hist, mask, gaps, cands, bucket, day = _batch(256, 1)
    for m, name in ((base, "loop"), (fast, "vector")):
        m.eval()
        for _ in range(2):
            with torch.no_grad():
                m(hist, mask, gaps, cands, bucket, day)
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad():
                m(hist, mask, gaps, cands, bucket, day)
        print(f"{name}_ms={(time.perf_counter()-t0)/5*1000:.1f}", flush=True)


def blend_auc(y, base, raw):
    rank = lambda v: rankdata(v, method="average") / (len(v) + 1)
    br, rr = rank(base), rank(raw)
    rows = []
    for w in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        pred = (1 - w) * br + w * rr
        rows.append((w, float(roc_auc_score(y, pred))))
    return max(rows, key=lambda r: r[1]), rows


def train(day: int, epochs: int, rank: int, batch: int):
    t0 = time.time()
    prefix = f"v21_fastdin_day{day}"
    if any(ROOT.glob(f"{prefix}*")):
        raise FileExistsError(prefix)
    val, y, events, base = load_v20(day)
    fit = fit_rows(day)
    print(f"packing fit={len(fit)} val={len(val)}", flush=True)
    td = pack_events(fit, hist, htimes)
    vd = pack_events(val, hist, htimes)
    model = FastDIN(NI, dim=64, rank=rank)
    ckpt = ROOT / f"v21_din_frozen_als_day{day}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.item.weight.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED + 100 + day)
    idx = np.where(td["has_click"])[0]
    print(f"fast DIN day{day} warm={ckpt.name} clicks={len(idx)} rank={rank}", flush=True)
    history = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        model.item.weight.requires_grad_(False)
        tot = n = 0.0
        for batch_i in event_batches(td, idx, bs=batch, shuffle=True, rng=rng):
            batch_i["mask"] = batch_i["mask"].bool()
            batch_i["has_click"] = batch_i["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(model, batch_i)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * batch_i["hist"].size(0)
            n += batch_i["hist"].size(0)
        raw = predict_probs(model, vd)
        auc = float(roc_auc_score(y, raw))
        (best_w, best_b), grid = blend_auc(y, base, raw)
        w20 = dict(grid)[0.2]
        row = {
            "epoch": epoch,
            "loss": tot / max(n, 1),
            "auc": auc,
            "blend_auc": best_b,
            "blend_w": best_w,
            "blend_auc_w20": w20,
            "v20_auc": float(roc_auc_score(y, base)),
            "prior_best_blend": BEST_BLEND[day],
            "delta_vs_prior_blend": best_b - BEST_BLEND[day],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), ROOT / f"{prefix}_ep{epoch}.pt")
        np.savez_compressed(
            ROOT / f"{prefix}_ep{epoch}_raw.npz", y=y, events=events, raw=raw, base=base
        )
        if best is None or best_b > best["blend_auc"]:
            best = row
            torch.save(model.state_dict(), ROOT / f"{prefix}.pt")
            np.savez_compressed(
                ROOT / f"{prefix}_raw.npz", y=y, events=events, raw=raw, base=base
            )
    report = {
        "day": day,
        "rank": rank,
        "epochs": epochs,
        "batch": batch,
        "warm_start": ckpt.name,
        "best": best,
        "history": history,
        "runtime_s": time.time() - t0,
        "beats_prior_blend": bool(best and best["blend_auc"] > BEST_BLEND[day]),
    }
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    append_log(
        f"V21-fastDIN-d{day}",
        f"Vectorized DIN + rank-{rank} adapter day{day}",
        BEST_BLEND[day],
        best["blend_auc"],
        report["runtime_s"],
        report["beats_prior_blend"],
        f"raw={best['auc']:.6f} w={best['blend_w']}",
    )
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--day", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.check:
        check_parity()
        return
    train(args.day, args.epochs, args.rank, args.batch)


if __name__ == "__main__":
    main()
