"""Multi-seed one-epoch fast DIN + optional low-LR second epoch + old-DIN member.

Day 6 selects the recipe; day 5 runs the frozen recipe. No submission written.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from v21_din import (
    NI,
    ROOT,
    SEED,
    batch_loss,
    event_batches,
    fit_rows,
    hist,
    htimes,
    load_v20,
    pack_events,
    predict_probs,
    append_log,
)
from v21_fast_din import BEST_BLEND, FastDIN

FAST1 = {6: 0.8239436671631495, 5: 0.8165425964775535}


def rank(v):
    return rankdata(v, method="average") / (len(v) + 1)


def run_epoch(model, opt, td, idx, rng, bs):
    model.train()
    model.item.weight.requires_grad_(False)
    tot = n = 0.0
    for b in event_batches(td, idx, bs=bs, shuffle=True, rng=rng):
        b["mask"] = b["mask"].bool()
        b["has_click"] = b["has_click"].bool()
        opt.zero_grad()
        loss, _ = batch_loss(model, b)
        loss.backward()
        opt.step()
        tot += float(loss.detach()) * b["hist"].size(0)
        n += b["hist"].size(0)
    return tot / max(n, 1)


def fit_seed(day, td, idx, seed, lr2, bs, vd=None):
    torch.manual_seed(seed)
    model = FastDIN(NI, dim=64, rank=16)
    state = torch.load(ROOT / f"v21_din_frozen_als_day{day}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.item.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    out = {}
    loss1 = run_epoch(model, opt, td, idx, rng, bs)
    out["ep1"] = (predict_probs(model, vd) if vd is not None else None, loss1, {k: v.clone() for k, v in model.state_dict().items()})
    if lr2:
        for g in opt.param_groups:
            g["lr"] = lr2
        loss2 = run_epoch(model, opt, td, idx, rng, bs)
        out["ep2"] = (predict_probs(model, vd) if vd is not None else None, loss2, {k: v.clone() for k, v in model.state_dict().items()})
    return out


def select(day, seeds, lr2, bs):
    t0 = time.time()
    prefix = f"v21_fastseeds_day{day}"
    if any(ROOT.glob(f"{prefix}*")):
        raise FileExistsError(prefix)
    val, y, events, base = load_v20(day)
    td = pack_events(fit_rows(day), hist, htimes)
    vd = pack_events(val, hist, htimes)
    idx = np.where(td["has_click"])[0]
    members = {}
    log = []
    for s in seeds:
        res = fit_seed(day, td, idx, SEED + 100 + day + 1000 * s, lr2, bs, vd)
        for tag, (raw, loss, state) in res.items():
            name = f"s{s}_{tag}"
            members[name] = raw
            auc = float(roc_auc_score(y, raw))
            b20 = float(roc_auc_score(y, 0.8 * rank(base) + 0.2 * rank(raw)))
            row = {"member": name, "loss": loss, "auc": auc, "blend_w20": b20}
            log.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(ROOT / f"{prefix}_{name}_raw.npz", y=y, events=events, raw=raw, base=base)
            torch.save(state, ROOT / f"{prefix}_{name}.pt")
    old = np.load(ROOT / f"v21_din_frozen_als_day{day}_raw.npz")["raw"]
    assert len(old) == len(y)
    members["old_ep8"] = old
    # Candidate composites
    ep1 = [members[f"s{s}_ep1"] for s in seeds]
    ep2 = [members[f"s{s}_ep2"] for s in seeds if f"s{s}_ep2" in members]
    composites = {
        "fast1_single": members[f"s{seeds[0]}_ep1"],
        "fast1_mean": np.mean(ep1, 0),
        "fast1_rankmean": np.mean([rank(r) for r in ep1], 0),
    }
    if ep2:
        composites["fast2_mean"] = np.mean(ep2, 0)
        composites["fast12_mean"] = np.mean(ep1 + ep2, 0)
    br = rank(base)
    ro = rank(old)
    results = []
    for name, raw in composites.items():
        rr = rank(raw)
        for wf, wo in itertools.product([0.15, 0.2, 0.25, 0.3], [0.0, 0.05, 0.1]):
            pred = (1 - wf - wo) * br + wf * rr + wo * ro
            auc = float(roc_auc_score(y, pred))
            results.append({"composite": name, "w_fast": wf, "w_old": wo, "auc": auc})
    results.sort(key=lambda r: -r["auc"])
    best = results[0]
    for r in results[:8]:
        print(json.dumps(r), flush=True)
    report = {
        "day": day,
        "seeds": seeds,
        "lr2": lr2,
        "batch": bs,
        "members": log,
        "best": best,
        "top": results[:15],
        "fast1_prior": FAST1[day],
        "delta_vs_fast1": best["auc"] - FAST1[day],
        "runtime_s": time.time() - t0,
    }
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    append_log(
        f"V21-fastseeds-d{day}",
        f"{len(seeds)} seeds fast DIN; best={best['composite']} wf={best['w_fast']} wo={best['w_old']}",
        FAST1[day],
        best["auc"],
        report["runtime_s"],
        best["auc"] > FAST1[day],
        f"n_members={len(log)}",
    )
    print(json.dumps({k: report[k] for k in report if k not in ("top", "members")}, indent=2), flush=True)


def confirm(day, bs):
    t0 = time.time()
    rec = json.loads((ROOT / "v21_fastseeds_day6_report.json").read_text())
    best = rec["best"]
    seeds = rec["seeds"]
    lr2 = rec["lr2"]
    prefix = f"v21_fastseeds_day{day}"
    if any(ROOT.glob(f"{prefix}*")):
        raise FileExistsError(prefix)
    val, y, events, base = load_v20(day)
    td = pack_events(fit_rows(day), hist, htimes)
    vd = pack_events(val, hist, htimes)
    idx = np.where(td["has_click"])[0]
    need_ep2 = best["composite"] in ("fast2_mean", "fast12_mean")
    use_seeds = seeds if best["composite"] != "fast1_single" else seeds[:1]
    members = {}
    for s in use_seeds:
        res = fit_seed(day, td, idx, SEED + 100 + day + 1000 * s, lr2 if need_ep2 else None, bs, vd)
        for tag, (raw, loss, state) in res.items():
            name = f"s{s}_{tag}"
            members[name] = raw
            print(json.dumps({"member": name, "loss": loss, "auc": float(roc_auc_score(y, raw))}), flush=True)
            np.savez_compressed(ROOT / f"{prefix}_{name}_raw.npz", y=y, events=events, raw=raw, base=base)
    ep1 = [members[f"s{s}_ep1"] for s in use_seeds]
    ep2 = [members[f"s{s}_ep2"] for s in use_seeds if f"s{s}_ep2" in members]
    comp = {
        "fast1_single": ep1[0],
        "fast1_mean": np.mean(ep1, 0),
        "fast1_rankmean": np.mean([rank(r) for r in ep1], 0),
        "fast2_mean": np.mean(ep2, 0) if ep2 else None,
        "fast12_mean": np.mean(ep1 + ep2, 0) if ep2 else None,
    }[best["composite"]]
    old = np.load(ROOT / f"v21_din_frozen_als_day{day}_raw.npz")["raw"]
    pred = (1 - best["w_fast"] - best["w_old"]) * rank(base) + best["w_fast"] * rank(comp) + best["w_old"] * rank(old)
    auc = float(roc_auc_score(y, pred))
    report = {
        "day": day,
        "recipe": best,
        "auc": auc,
        "fast1_prior": FAST1[day],
        "delta_vs_fast1": auc - FAST1[day],
        "v20_auc": float(roc_auc_score(y, base)),
        "kept": auc > FAST1[day],
        "runtime_s": time.time() - t0,
    }
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    append_log(
        f"V21-fastseeds-d{day}",
        f"Confirm frozen recipe {best['composite']} wf={best['w_fast']} wo={best['w_old']}",
        FAST1[day],
        auc,
        report["runtime_s"],
        report["kept"],
        "",
    )
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr2", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.confirm:
        confirm(a.day, a.batch)
    else:
        select(a.day, list(range(a.seeds)), a.lr2 if a.lr2 > 0 else None, a.batch)


if __name__ == "__main__":
    main()
