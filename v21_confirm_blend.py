"""Freeze the primary-date DIN recipe, then audit its untouched-date confirmation.

This script does not select weights on the confirmation date or export a CSV.
"""
import argparse
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as ROOT

MANIFEST = ROOT / "v21_als6_blend_frozen.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(day, member, weight):
    with np.load(ROOT / f"imp19_tree_day{day}.npz") as base, np.load(member) as candidate:
        for key in ("y", "events"):
            if not np.array_equal(base[key], candidate[key]):
                raise ValueError(f"Cohort mismatch: {key}, day {day}")
        if not np.array_equal(base["pred"], candidate["base"]):
            raise ValueError("Candidate was produced against a different baseline")
        y = base["y"]
        raw = candidate["raw"]
        if raw.shape != y.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid candidate predictions")
        rank = lambda values: rankdata(values, method="average") / (len(values) + 1)
        mixed = (1 - weight) * rank(base["pred"]) + weight * rank(raw)
        old = float(roc_auc_score(y, base["pred"]))
        new = float(roc_auc_score(y, mixed))
        return {"baseline_auc": old, "blend_auc": new, "gain": new - old,
                "raw_auc": float(roc_auc_score(y, raw)), "rows": len(y)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        member = ROOT / "v21_din_frozen_als_day6_ep6_raw.npz"
        sources = [member, ROOT / "v21_din_frozen_als_day6_ep6.pt",
                   ROOT / "imp19_tree_day6.npz", ROOT / "imp19_tree_day5.npz",
                   ROOT / "imp17_als_cut3.npz", ROOT / "imp17_als_cut4.npz"]
        report = {"epoch": 6, "weight": 0.2, "init": "als", "dim": 64,
                  "lr": 0.003, "seed": 42, "fit_days": 3,
                  "primary": score(6, member, 0.2),
                  "source_sha256": {p.name: digest(p) for p in sources},
                  "confirmation_tag": "v21_als6_confirm_day5",
                  "status": "primary_only_not_promoted"}
        with MANIFEST.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    else:
        report = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, expected in report["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        tag = report["confirmation_tag"]
        training = json.loads((ROOT / f"{tag}_report.json").read_text())
        if not training["fixed_epoch_confirmation"] or training["best_epoch"] != report["epoch"]:
            raise ValueError("Confirmation selected a different epoch")
        for key in ("dim", "lr", "seed"):
            if training[key] != report[key]:
                raise ValueError(f"Confirmation recipe mismatch: {key}")
        report["confirmation"] = score(5, ROOT / f"{tag}_raw.npz", report["weight"])
        report["status"] = "both_dates_positive_pending_independent_audit" if (
            report["primary"]["gain"] > 0 and report["confirmation"]["gain"] > 0
        ) else "rejected"
        (ROOT / "v21_als6_confirmation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
