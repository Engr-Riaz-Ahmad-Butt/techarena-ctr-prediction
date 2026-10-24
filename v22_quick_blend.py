"""Small primary-date temperature/blend search on saved predictions; no fitting.

Freeze the primary winner before loading day5. Do not export a submission.
"""
import json
import hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as ROOT
from tst2_candidate import influence


def blend(base, raw, temperature, weight):
    p = np.asarray(raw, dtype=np.float64).reshape(-1, 10)
    if temperature != 1:
        logits = np.log(np.maximum(p, 1e-30)) / temperature
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
    rank = lambda x: rankdata(x, method="average") / (len(x) + 1)
    return (1 - weight) * rank(base) + weight * rank(p.ravel())


def load(day):
    with np.load(ROOT / f"imp19_tree_day{day}.npz") as b, np.load(ROOT / f"v21_fastdin_day{day}_raw.npz") as n:
        for key in ("y", "events"):
            if not np.array_equal(b[key], n[key]):
                raise ValueError(f"Cohort mismatch {key}")
        if not np.array_equal(b["pred"], n["base"]):
            raise ValueError("Baseline mismatch")
        y, events, base, raw = b["y"], b["events"], b["pred"], n["raw"]
    assert raw.shape == y.shape and np.isfinite(raw).all() and (raw >= 0).all()
    assert np.allclose(raw.reshape(-1, 10).sum(1), 1, atol=1e-6)
    return y, events, base, raw


def evaluate(day, recipe):
    y, events, base, raw = load(day)
    old = blend(base, raw, 1, .2)
    pred = blend(base, raw, recipe["temperature"], recipe["weight"])
    previous, current = float(roc_auc_score(y, old)), float(roc_auc_score(y, pred))
    delta = current - previous
    se = float(np.linalg.norm(influence(y, pred) - influence(y, old)))
    np.savez_compressed(ROOT / f"v22_quick_day{day}.npz", y=y, events=events, pred=pred)
    return {"day": day, "previous_auc": previous, "auc": current, "gain": delta,
            "conditional_paired_interval": [delta - 1.96 * se, delta + 1.96 * se]}


def main():
    frozen = ROOT / "v22_quick_frozen.json"
    if frozen.exists():
        raise FileExistsError("Existing experiment must not be silently retuned")
    y, events, base, raw = load(6)
    trials = []
    for temperature in [.7, 1., 1.3]:
        for weight in [.16, .18, .20, .22, .24]:
            pred = blend(base, raw, temperature, weight)
            trials.append({"temperature": temperature, "weight": weight,
                           "auc": float(roc_auc_score(y, pred))})
    selected = max(trials, key=lambda x: x["auc"])
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for day in (6, 5) for name in (f"imp19_tree_day{day}.npz", f"v21_fastdin_day{day}_raw.npz")}
    frozen.open("x").write(json.dumps({"selected": selected, "trials": trials, "input_sha256": hashes}, indent=2))
    reports = [evaluate(day, selected) for day in (6, 5)]
    passed = all(r["gain"] > 0 and r["conditional_paired_interval"][0] > 0 for r in reports)
    report = {"selected": selected, "reports": reports, "both_date_conditional_gate": passed,
              "status": "candidate_pending_recipe_audit" if passed else "not_promoted",
              "note": "Exploratory reused validation; intervals exclude selection uncertainty. No official-score or CSV-readiness claim."}
    (ROOT / "v22_quick_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
