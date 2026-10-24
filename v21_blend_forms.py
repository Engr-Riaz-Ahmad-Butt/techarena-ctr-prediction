"""Primary-date-only comparison of complementary DIN blend forms.

Confirmation uses the primary winner unchanged, with no search on day5.
"""
import argparse
import json
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from config import ARTIFACT_DIR as ROOT
from v21_confirm_blend import MANIFEST, digest


def combine(base, raw, method, weight):
    if method == "rank":
        rank = lambda x: rankdata(x, method="average") / (len(x) + 1)
        return (1 - weight) * rank(base) + weight * rank(raw)
    if method == "logit":
        return logit(np.clip(base, 1e-6, 1 - 1e-6)) + weight * np.log(np.maximum(raw, 1e-12))
    if method == "conditional":
        b = base.reshape(-1, 10)
        mass = b.sum(axis=1, keepdims=True)
        return ((1 - weight) * b + weight * mass * raw.reshape(-1, 10)).ravel()
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    frozen_path = ROOT / "v21_blend_forms_frozen.json"
    day = 5 if args.confirm else 6
    frozen = json.loads(MANIFEST.read_text())
    for name, expected in frozen["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    member = ROOT / ("v21_als6_confirm_day5_raw.npz" if args.confirm else
                     "v21_din_frozen_als_day6_ep6_raw.npz")
    with np.load(ROOT / f"imp19_tree_day{day}.npz") as ref, np.load(member) as cand:
        for key in ("y", "events"):
            if not np.array_equal(ref[key], cand[key]):
                raise ValueError(f"Cohort mismatch: {key}")
        if not np.array_equal(ref["pred"], cand["base"]):
            raise ValueError("Baseline mismatch")
        base, raw, y = ref["pred"].astype(float), cand["raw"].astype(float), ref["y"]
    if args.confirm:
        selected = json.loads(frozen_path.read_text())
        recipe = selected["selected"]
        training = json.loads((ROOT / "v21_als6_confirm_day5_report.json").read_text())
        if not (training["day"] == 5 and training["init"] == "als" and
                training["fixed_epoch_confirmation"] and training["best_epoch"] == 6):
            raise ValueError("Confirmation recipe mismatch")
        selected["confirmation_auc"] = float(roc_auc_score(y, combine(base, raw, recipe["method"], recipe["weight"])))
        selected["confirmation_baseline_auc"] = float(roc_auc_score(y, base))
        selected["status"] = "pending_independent_audit"
        (ROOT / "v21_blend_forms_confirmation.json").write_text(json.dumps(selected, indent=2))
        print(json.dumps(selected, indent=2))
        return
    trials = []
    for method, weights in (("rank", [.2]), ("logit", [.05, .1, .2, .3, .5, .75, 1]),
                            ("conditional", [.05, .1, .2, .3, .5])):
        for weight in weights:
            auc = float(roc_auc_score(y, combine(base, raw, method, weight)))
            trials.append({"method": method, "weight": weight, "auc": auc})
    report = {"selected": max(trials, key=lambda row: row["auc"]), "trials": trials,
              "status": "exploratory_primary_only", "epoch": 6,
              "note": "Repeated primary-date selection; no 0.84 or official-score claim."}
    with frozen_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
