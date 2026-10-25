"""Explicit history-candidate interaction residual, warm-started from V23.

Primary-date blend selection is frozen before day5. Existing validation dates
have been reused; this experiment does not claim an untouched holdout score.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as ROOT
from v21_din import NI, SEED, fit_rows, load_v20, pack_events, hist, htimes, event_batches
from v21_fast_din import FastDIN
from tst2_candidate import influence


class OutsideDIN(FastDIN):
    def __init__(self, n_items, dim=64, rank=16):
        super().__init__(n_items, dim=dim, rank=rank)
        self.outside = torch.nn.Parameter(torch.zeros(()))
        self.interaction = torch.nn.Linear(dim*2+3, 1)
        torch.nn.init.zeros_(self.interaction.weight)
        torch.nn.init.zeros_(self.interaction.bias)

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
        hunit = torch.nn.functional.normalize(he, dim=-1)
        cunit = torch.nn.functional.normalize(ce, dim=-1)
        context_unit = torch.nn.functional.normalize(ctx, dim=-1)
        similarities = (hunit[:, None, :, :] * cunit[:, :, None, :]).sum(-1)
        masked = similarities.masked_fill(~mask[:, None, :], -1)
        maximum = masked.max(-1).values.masked_fill(all_pad[:, None], 0)
        recent = similarities[:, :, -1].masked_fill(all_pad[:, None], 0)
        cosine = (context_unit * cunit).sum(-1)
        direct = torch.cat([context_unit*cunit, torch.abs(context_unit-cunit),
                            cosine.unsqueeze(-1), maximum.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        return self.mlp(feat).squeeze(-1) + self.interaction(direct).squeeze(-1)

    def full_logits(self, batch):
        logits = self(batch["hist"], batch["mask"].bool(), batch["gaps"],
                      batch["cands"], batch["bucket"], batch["day_type"])
        return torch.cat([logits, self.outside.expand(len(logits), 1)], dim=1)


def rank(values):
    return rankdata(values, method="average") / (len(values) + 1)


def baseline(day, y, events, base):
    ranks = []
    for seed in (0, 1, 2):
        with np.load(ROOT / f"v21_fastseeds_day{day}_s{seed}_ep1_raw.npz") as source:
            assert np.array_equal(y, source["y"]) and np.array_equal(events, source["events"])
            assert np.array_equal(base, source["base"])
            ranks.append(rank(source["raw"]))
    return .8 * rank(base) + .2 * np.mean(ranks, axis=0)


def predict(model, data):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in event_batches(data, np.arange(data["n"]), bs=512, shuffle=False):
            output.append(torch.softmax(model.full_logits(batch), dim=1)[:, :10].numpy())
    return np.concatenate(output).reshape(-1)


def progress(**fields):
    fields["unix_time"] = time.time()
    path = ROOT / "v27_interaction_progress.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, indent=2))
    tmp.replace(path)
    print(json.dumps(fields), flush=True)


def train_day(day, chosen=None):
    prefix = f"v27_interaction_day{day}"
    if any(ROOT.glob(f"{prefix}*")):
        raise FileExistsError(prefix)
    start = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    val, y, events, base = load_v20(day)
    old = baseline(day, y, events, base)
    expected = {6: .8241639495695283, 5: .8166261127941717}
    assert abs(roc_auc_score(y, old) - expected[day]) < 1e-10
    reference = np.load(ROOT / f"v23_outside_day{day}.npz")["pred"]
    td = pack_events(fit_rows(day), hist, htimes)
    vd = pack_events(val, hist, htimes)
    model = OutsideDIN(NI)
    source = ROOT / f"v23_outside_day{day}.pt"
    loaded = model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=False)
    assert set(loaded.missing_keys) == {"interaction.weight", "interaction.bias"} and not loaded.unexpected_keys
    model.item.weight.requires_grad_(False)
    click_rate = float(td["has_click"].mean())
    initial = float(model.outside.detach())
    optimizer = torch.optim.Adam([
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("interaction.")], "lr": .0001, "interaction_lr": .003},
        {"params": model.interaction.parameters(), "lr": .003}], weight_decay=1e-5)
    indexes = np.arange(td["n"])
    total = (len(indexes) + 511) // 512
    rng = np.random.default_rng(SEED + 300 + day)
    model.train()
    progress(stage="training", day=day, batches=total, threads=4, initial_outside=initial)
    for i, batch in enumerate(event_batches(td, indexes, bs=512, shuffle=True, rng=rng), 1):
        targets = torch.where(batch["has_click"].bool(), batch["target"], 10)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model.full_logits(batch), targets)
        loss.backward()
        optimizer.step()
        if i % 100 == 0 or i == total:
            progress(stage="training", day=day, batch=i, batches=total, elapsed_s=time.monotonic() - start)
    torch.save(model.state_dict(), ROOT / f"{prefix}.pt")
    raw = predict(model, vd)
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw.reshape(-1, 10).sum(1) <= 1.000001).all()
    np.savez_compressed(ROOT / f"{prefix}_raw.npz", y=y, events=events, raw=raw, base=old)
    previous = float(roc_auc_score(y, reference))
    trials = []
    for weight in ([chosen["weight"]] if chosen else [.1]):
        pred = (1 - weight) * rank(old) + weight * rank(raw)
        auc = float(roc_auc_score(y, pred))
        trials.append({"weight": weight, "auc": auc})
    selected = max(trials, key=lambda r: r["auc"])
    pred = (1 - selected["weight"]) * rank(old) + selected["weight"] * rank(raw)
    gain = selected["auc"] - previous
    se = float(np.linalg.norm(influence(y, pred) - influence(y, reference)))
    selected.update(day=day, previous_auc=previous, gain=gain, raw_auc=float(roc_auc_score(y, raw)),
                    conditional_interval=[gain - 1.96 * se, gain + 1.96 * se])
    report = {"selected": selected, "trials": trials, "epochs": 1, "lr": .0001, "interaction_lr": .003,
              "training_click_rate": click_rate, "initial_outside": initial,
              "elapsed_s": time.monotonic() - start}
    (ROOT / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return selected


def main():
    inputs = [ROOT / f"v23_outside_day{d}.pt" for d in (6, 5)]
    inputs += [ROOT / f"v23_outside_day{d}.npz" for d in (6, 5)]
    inputs += [ROOT / f"imp19_tree_day{d}.npz" for d in (6, 5)]
    inputs += [ROOT / f"v21_fastseeds_day{d}_s{s}_ep1_raw.npz" for d in (6, 5) for s in (0, 1, 2)]
    sources = [Path(__file__), Path("v21_din.py"), Path("v21_fast_din.py")]
    frozen = {"epochs": 1, "lr": .0001, "interaction_lr": .003, "seed": SEED, "batch": 512, "threads": 4,
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (ROOT / "v27_interaction_frozen.json").open("x").write(json.dumps(frozen, indent=2))
    selected = train_day(6)
    (ROOT / "v27_interaction_selection.json").open("x").write(json.dumps(selected, indent=2))
    reports = [selected]
    if selected["gain"] > 0 and selected["conditional_interval"][0] > 0:
        for p in inputs:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == frozen["input_sha256"][p.name]
        reports.append(train_day(5, selected))
    passed = len(reports) == 2 and all(r["conditional_interval"][0] > 0 for r in reports)
    result = {"reports": reports, "both_date_conditional_gate": passed,
              "status": "candidate_pending_independent_audit" if passed else "not_promoted",
              "note": "Reused local validation, no official-score or CSV-readiness claim."}
    (ROOT / "v27_interaction_report.json").write_text(json.dumps(result, indent=2))
    progress(stage=result["status"], reports=reports)


if __name__ == "__main__":
    main()
