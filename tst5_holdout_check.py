"""tst5: fresh-holdout check vs nxa_h1. RUN ONLY for a candidate that passed the +0.0005 gate vs nxa_h1 on the day-6 selection
subset AND day 5 with the selection grid verified (selection-subset only). Never used for selection or tuning.

Usage: python tst5_holdout_check.py NAME DAY6_FULL_NPZ OUT_JSON   (candidate npz key 'pred' over the 351,380 day-6 rows, V53 order)
Holdout: artifacts/fresh_holdout_day6.npz (10,716 user-disjoint day-6 events; weights = holdout-file weights).
Pass rule (protocol_v4): adjusted gain vs nxa_h1 >= 0. Paired event bootstrap 1000 resamples, seed 3. Also vs V53 / V39.
Note: the holdout was already scored for nxa_h1 (2026-10-29), so it is no longer pristine.
"""
import json, sys
from pathlib import Path
import numpy as np

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
sys.path.insert(0, str(B))
from tst5_candidate_eval import paired_many


def main():
    name, path, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    ho = np.load(R / "fresh_holdout_day6.npz"); z = np.load(path); v = np.load(R / "v53_causal_propensity_day6.npz"); h = np.load(R / "nxa_h1_day6.npz"); v39 = np.load(R / "v39_regularized_als32_day6.npz")
    fi = ho["full_indices"]
    assert np.array_equal(z["y"], v["y"]) and np.array_equal(z["events"], v["events"]) and len(z["pred"]) == 351380
    y = ho["y"]; e = ho["events"]; w = ho["weights"]; assert np.array_equal(y, v["y"][fi]) and np.array_equal(ho["baseline"], v["pred"][fi])
    sel = np.load(R / "selection_day6.npz"); assert len(np.intersect1d(sel["full_indices"], fi)) == 0
    arms = dict(cand=(z["pred"][fi], w), h1=(h["pred"][fi], w), v53=(v["pred"][fi], w), v39=(v39["pred"][fi], w), h1_v39=(h["pred_v39"][fi], w),
                cand_plain=(z["pred"][fi], np.ones(len(y))), h1_plain=(h["pred"][fi], np.ones(len(y))))
    c39 = "cand"
    if "pred_v39" in z.files: arms["cand_v39"] = (z["pred_v39"][fi], w); c39 = "cand_v39"
    pairs = dict(vs_h1_adjusted=("h1", "cand"), vs_h1_plain=("h1_plain", "cand_plain"), vs_v53_adjusted=("v53", "cand"), v39_parent_vs_v39_adjusted=("v39", c39), v39_parent_vs_h1_v39_adjusted=("h1_v39", c39))
    if "raw" in z.files: arms["raw"] = (np.asarray(z["raw"], dtype="float64")[fi], w); pairs["raw_vs_h1_adjusted"] = ("h1", "raw")
    out = dict(name=name, holdout_events=int(len(np.unique(e))), holdout_rows=int(len(y)), disjoint_from_selection_subset=True, **paired_many(y, e, arms, pairs, 1000))
    out["holdout_passed"] = bool(out["vs_h1_adjusted"]["gain"] >= 0.0)
    out["holdout_note"] = "holdout previously scored for nxa_h1 (2026-10-29 22:11); repeated use reduces its independence"
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: ({kk: v_[kk] for kk in ("base", "candidate", "gain", "paired95")} if isinstance(v_, dict) else v_) for k, v_ in out.items()}, indent=1))


if __name__ == "__main__":
    main()
