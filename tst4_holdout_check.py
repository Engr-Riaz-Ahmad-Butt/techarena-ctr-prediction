"""tst4: fresh-holdout check. RUN ONLY for a candidate that (a) passed the +0.0005 gate on both days and (b) was selected on
the day-6 SELECTION subset only (verified from its saved selection grid). Never used for selection or tuning.

Usage: python tst4_holdout_check.py NAME DAY6_FULL_NPZ OUT_JSON   (candidate npz key 'pred' over the full 351,380 day-6 rows)
Holdout: artifacts/fresh_holdout_day6.npz (10,716 user-disjoint day-6 events; keys y, events, baseline (= V53), weights, full_indices).
Pass rule (protocol_v3): adjusted gain vs V53 >= 0 on the holdout. Paired event bootstrap 1000 resamples, seed 3, reported.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
sys.path.insert(0, str(B))
from tst4_candidate_eval import paired  # same paired bootstrap (1000, seed 3)


def main():
    name, path, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    ho = np.load(R / "fresh_holdout_day6.npz"); z = np.load(path); v = np.load(R / "v53_causal_propensity_day6.npz"); fi = ho["full_indices"]
    assert np.array_equal(z["y"], v["y"]) and np.array_equal(z["events"], v["events"]) and len(z["pred"]) == 351380
    y = ho["y"]; e = ho["events"]; base = ho["baseline"]; cand = z["pred"][fi]; w = ho["weights"]
    assert np.array_equal(y, v["y"][fi]) and np.array_equal(base, v["pred"][fi])
    v39 = np.load(R / "v39_regularized_als32_day6.npz"); p39 = (z["pred_v39"] if "pred_v39" in z.files else z["pred"])[fi]
    out = dict(name=name, holdout_events=int(len(np.unique(e))), holdout_rows=int(len(y)),
               vs_v53_adjusted=paired(y, base, cand, w, e), vs_v53_plain=paired(y, base, cand, np.ones(len(y)), e),
               vs_v39_adjusted=paired(y, v39["pred"][fi], p39, w, e))
    out["holdout_passed"] = bool(out["vs_v53_adjusted"]["gain"] >= 0.0)
    if len(sys.argv) > 5:  # optional current baseline: argv[4] = name, argv[5] = its day-6 npz (key 'pred', full 351,380 rows)
        xb = np.load(sys.argv[5]); assert np.array_equal(xb["y"], v["y"]) and np.array_equal(xb["events"], v["events"])
        out["vs_" + sys.argv[4] + "_adjusted"] = paired(y, xb["pred"][fi], cand, w, e); out["holdout_passed"] = bool(out["vs_" + sys.argv[4] + "_adjusted"]["gain"] >= 0.0)
        out["holdout_note"] = "holdout previously scored once (nxa_h1 acceptance, 2026-10-29 22:11); repeated use reduces its independence"
    outp.write_text(json.dumps(out, indent=2)); print(json.dumps({k: (v_ if not isinstance(v_, dict) else {kk: v_[kk] for kk in ("v53", "candidate", "gain", "paired95")}) for k, v_ in out.items()}, indent=1))


if __name__ == "__main__":
    main()
