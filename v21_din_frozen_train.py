"""
Proper DIN with FROZEN item embeddings (SVD or ALS), trainable attention+MLP.
Saves softmax probabilities. Compares inits; confirms on day 5.
"""
from __future__ import annotations

import argparse
import json
import time

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
    V20,
    append_log,
    batch_loss,
    event_batches,
    fit_rows,
    hist,
    htimes,
    init_embeddings,
    load_als_item_factors,
    load_v20,
    pack_events,
    predict_probs,
)
from v21_experiments import item_embeddings

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(6)


def train_frozen(day: int, init: str, dim: int = 64, max_epochs: int = 12, lr: float = 3e-3,
                 fixed_epoch: bool = False, tag: str | None = None):
    t0 = time.time()
    prefix = tag or f"v21_din_frozen_{init}_day{day}"
    if fixed_epoch and not tag:
        raise ValueError("Confirmation requires a separate output tag")
    if tag and any(p.suffix in {".pt", ".npz", ".json"} for p in ROOT.glob(f"{prefix}*")):
        raise FileExistsError(f"Outputs already exist for {prefix}")
    val, y, events, base = load_v20(day)
    fit = fit_rows(day)
    td = pack_events(fit, hist, htimes)
    vd = pack_events(val, hist, htimes)
    model = DIN(NI, dim=dim)
    if init == "svd":
        emb, _ = item_embeddings(START + (day - 2) * DAY, dim=dim)
        with torch.no_grad():
            model.item.weight[:-1].copy_(torch.from_numpy(emb))
            model.item.weight[-1].zero_()
    else:
        init_embeddings(model, "als", als_cut=max(day - 2, 2), dim=dim)
    model.item.weight.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(SEED + day)
    idx = np.where(td["has_click"])[0]
    print(f"frozen DIN day{day} init={init} clicks={len(idx)}", flush=True)
    best_auc, best_raw, best_ep, best_state = -1.0, None, 0, None
    history = []
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        model.item.weight.requires_grad_(False)
        tot = n = 0.0
        for batch in event_batches(td, idx, bs=256, shuffle=True, rng=rng):
            batch["mask"] = batch["mask"].bool()
            batch["has_click"] = batch["has_click"].bool()
            opt.zero_grad()
            loss, _ = batch_loss(model, batch)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * batch["hist"].size(0)
            n += batch["hist"].size(0)
        raw = predict_probs(model, vd)
        auc = float(roc_auc_score(y, raw))
        acc = float((raw.reshape(-1, 10).argmax(1) == vd["target"])[vd["has_click"]].mean())
        row = {"epoch": epoch, "loss": tot / max(n, 1), "auc": auc, "click_acc": acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        # checkpoint every epoch so a stall does not lose progress
        torch.save(model.state_dict(), ROOT / f"{prefix}_ep{epoch}.pt")
        np.savez_compressed(
            ROOT / f"{prefix}_ep{epoch}_raw.npz",
            y=y,
            events=events,
            raw=raw,
            base=base,
        )
        if fixed_epoch or auc > best_auc + 1e-5:
            best_auc, best_raw, best_ep = auc, raw, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ROOT / f"{prefix}.pt")
            np.savez_compressed(
                ROOT / f"{prefix}_raw.npz",
                y=y,
                events=events,
                raw=best_raw,
                base=base,
            )
            bad = 0
        else:
            bad += 1
            if bad >= 3:
                print("early stop", flush=True)
                break
    if best_state is None:
        raise RuntimeError("No successful epoch")
    torch.save(best_state, ROOT / f"{prefix}.pt")
    np.savez_compressed(
        ROOT / f"{prefix}_raw.npz",
        y=y,
        events=events,
        raw=best_raw,
        base=base,
    )
    report = {
        "day": day,
        "init": init,
        "raw_auc": best_auc,
        "best_epoch": best_ep,
        "fixed_epoch_confirmation": fixed_epoch,
        "dim": dim,
        "lr": lr,
        "seed": SEED,
        "v20_auc": float(roc_auc_score(y, base)),
        "history": history,
        "runtime_s": time.time() - t0,
    }
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    append_log(
        f"V21-DIN-frz-{init}-d{day}",
        f"Frozen-emb DIN init={init} day{day}",
        report["v20_auc"],
        best_auc,
        report["runtime_s"],
        False,
        f"epoch={best_ep}",
    )
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["svd", "als"], required=True)
    ap.add_argument("--day", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--fixed-epoch", action="store_true", help="Use exactly --epochs; never select on confirmation labels")
    ap.add_argument("--tag", help="Separate output prefix; existing outputs are refused")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    train_frozen(args.day, args.init, max_epochs=args.epochs,
                 fixed_epoch=args.fixed_epoch, tag=args.tag)


if __name__ == "__main__":
    main()
