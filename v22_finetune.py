"""One small-learning-rate epoch per date; primary-selected mixture frozen for day5."""
import json
import time
import hashlib
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from v21_fast_din import FastDIN
from v21_din import ROOT, NI, SEED, fit_rows, load_v20, pack_events, hist, htimes, event_batches, batch_loss, predict_probs
from tst2_candidate import influence


def rank(x):
    return rankdata(x, method="average") / (len(x) + 1)


def train(day, selected=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    prefix = f"v22_low_lr_day{day}"
    if any(ROOT.glob(f"{prefix}*")):
        raise FileExistsError(prefix)
    start = time.monotonic()
    rows, y, events, base = load_v20(day)
    td = pack_events(fit_rows(day), hist, htimes)
    vd = pack_events(rows, hist, htimes)
    with np.load(ROOT / f"v21_fastdin_day{day}_raw.npz") as prior:
        assert np.array_equal(y, prior["y"]) and np.array_equal(events, prior["events"])
        old_raw = prior["raw"].copy()
    model = FastDIN(NI, dim=64, rank=16)
    model.load_state_dict(torch.load(ROOT / f"v21_fastdin_day{day}.pt", map_location="cpu", weights_only=True))
    model.item.weight.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0003, weight_decay=1e-5)
    rng = np.random.default_rng(SEED + 200 + day)
    idx = np.where(td["has_click"])[0]
    total_batches = (len(idx) + 511) // 512
    model.train()
    print(json.dumps({"day": day, "batches": total_batches, "threads": 4, "epochs": 1, "lr": .0003}), flush=True)
    for i, batch in enumerate(event_batches(td, idx, bs=512, shuffle=True, rng=rng), 1):
        batch["mask"] = batch["mask"].bool()
        batch["has_click"] = batch["has_click"].bool()
        optimizer.zero_grad()
        loss, _ = batch_loss(model, batch)
        loss.backward()
        optimizer.step()
        if i % 50 == 0 or i == total_batches:
            print(json.dumps({"day": day, "batch": i, "total": total_batches,
                              "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
    torch.save(model.state_dict(), ROOT / f"{prefix}.pt")
    raw = predict_probs(model, vd)
    np.savez_compressed(ROOT / f"{prefix}_raw.npz", y=y, events=events, raw=raw, base=base)
    old = .8 * rank(base) + .2 * rank(old_raw)
    previous = float(roc_auc_score(y, old))
    trials = []
    for mixture in ([selected["mixture"]] if selected else [.25, .5, 1.]):
        mixed = (1 - mixture) * old_raw.astype(float) + mixture * raw.astype(float)
        pred = .8 * rank(base) + .2 * rank(mixed)
        auc = float(roc_auc_score(y, pred))
        se = float(np.linalg.norm(influence(y, pred) - influence(y, old)))
        trials.append({"day": day, "mixture": mixture, "auc": auc, "previous_auc": previous,
                       "gain": auc - previous, "conditional_interval": [auc - previous - 1.96 * se, auc - previous + 1.96 * se]})
    best = max(trials, key=lambda row: row["auc"])
    report = {"selected": best, "trials": trials, "elapsed_s": time.monotonic() - start,
              "note": "Exploratory local validation; no official-score or submission-readiness claim."}
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return best


if __name__ == "__main__":
    manifest = ROOT / "v22_low_lr_frozen.json"
    sources = [ROOT / f"v21_fastdin_day{day}{suffix}" for day in (6, 5) for suffix in (".pt", "_raw.npz")]
    manifest.open("x").write(json.dumps({"lr": .0003, "epochs": 1, "threads": 4, "batch": 512,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    chosen = train(6)
    (ROOT / "v22_low_lr_selected.json").open("x").write(json.dumps(chosen, indent=2))
    reports = [chosen]
    if chosen["gain"] > 0 and chosen["conditional_interval"][0] > 0:
        reports.append(train(5, chosen))
    passed = len(reports) == 2 and all(r["gain"] > 0 and r["conditional_interval"][0] > 0 for r in reports)
    result = {"reports": reports, "both_date_conditional_gate": passed,
              "status": "candidate_pending_independent_audit" if passed else "not_promoted"}
    (ROOT / "v22_low_lr_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
