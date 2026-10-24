"""Test small additions of four existing complementary experts, without fitting."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from v22_finetune import rank
from v22_quick_blend import load
from config import ARTIFACT_DIR as ROOT
from tst2_candidate import influence


def inputs(day, prefix):
    y, events, base, raw = load(day)
    current = .8 * rank(base) + .2 * rank(raw)
    with np.load(ROOT / f"{prefix}_day{day}.npz") as expert:
        assert np.array_equal(expert["y"], y) and np.array_equal(expert["events"], events)
        extra = expert["raw"] if "raw" in expert.files else expert["pred"]
    assert np.isfinite(extra).all() and extra.shape == current.shape
    return y, current, extra


def metric(day, prefix, weight):
    y, old, extra = inputs(day, prefix)
    pred = (1 - weight) * rank(old) + weight * rank(extra)
    previous, score = float(roc_auc_score(y, old)), float(roc_auc_score(y, pred))
    se = float(np.linalg.norm(influence(y, pred) - influence(y, old)))
    return {"day": day, "prefix": prefix, "weight": weight, "auc": score,
            "previous_auc": previous, "gain": score - previous,
            "conditional_interval": [score - previous - 1.96 * se, score - previous + 1.96 * se]}


if __name__ == "__main__":
    frozen = ROOT / "v22_diversity_frozen.json"
    if frozen.exists():
        raise FileExistsError(frozen)
    trials = []
    for prefix in ["v21_recency_raw", "v21_cooccur_raw", "v21_sequence_raw", "imp15_als"]:
        if not (ROOT / f"{prefix}_day5.npz").exists():
            continue
        y, old, extra = inputs(6, prefix)
        for weight in [.005, .01, .02, .05]:
            score = float(roc_auc_score(y, (1 - weight) * rank(old) + weight * rank(extra)))
            trials.append({"prefix": prefix, "weight": weight, "auc": score})
    best = max(trials, key=lambda x: x["auc"])
    frozen.open("x").write(json.dumps({"selected": best, "trials": trials}, indent=2))
    reports = [metric(6, best["prefix"], best["weight"])]
    if reports[0]["conditional_interval"][0] > 0:
        reports.append(metric(5, best["prefix"], best["weight"]))
    passed = len(reports) == 2 and all(r["conditional_interval"][0] > 0 for r in reports)
    result = {"reports": reports, "both_date_conditional_gate": passed,
              "status": "candidate_pending_audit" if passed else "not_promoted",
              "note": "Exploratory reused validation; no submission generated."}
    (ROOT / "v22_diversity_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
