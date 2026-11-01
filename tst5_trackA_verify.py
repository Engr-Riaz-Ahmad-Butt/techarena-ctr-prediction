"""tst5: verify the improver's Track A diagnostic (nxb_journal.md, 2026-10-30 ~12:15) from saved arrays only. No fitting.

Checks: a1/a2 standalone adjusted/plain AUC (day 6 full, selection subset), per-history-bin plain AUC vs nxa_h1/V53,
the reported best blend (rank 0.02 with nxa_h1, chosen on the selection subset) recomputed independently with a paired
event bootstrap vs nxa_h1, single-feature plain AUCs from artifacts/nxb_a_feat_k6.npy mapped to the cohort rows, and the
warm16 reference (artifacts/features/imp3_wf_val{5,6}.npy) quoted in the journal.
Usage: python tst5_trackA_verify.py OUT_JSON [BOOTS]
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
sys.path.insert(0, str(B))
from tst5_candidate_eval import weights5, hist_bins, paired_many


def rank(x): return rankdata(x, method="average") / (len(x) + 1)


def main():
    outp = Path(sys.argv[1]); boots = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; out = {}
    v = np.load(R / "v53_causal_propensity_day6.npz"); h = np.load(R / "nxa_h1_day6.npz"); s = np.load(R / "selection_day6.npz"); fi = s["full_indices"]
    y = v["y"]; e = v["events"]; w5 = weights5(e); bins = hist_bins(e)
    for name in ("a1", "a2"):
        z = np.load(R / f"nxb_a_{name}_day6.npz"); assert np.array_equal(z["y"], y) and np.array_equal(z["events"], e); r = z["raw"]
        out[name] = dict(full_adj=float(roc_auc_score(y, r, sample_weight=w5)), sel_adj=float(roc_auc_score(y[fi], r[fi], sample_weight=s["weights"])), plain=float(roc_auc_score(y, r)),
                         pred_equals_raw=bool(np.array_equal(z["pred"], r)),
                         bins_plain={b: dict(events=int((bins == b).sum() // 10), clean=float(roc_auc_score(y[bins == b], r[bins == b])), nxa_h1=float(roc_auc_score(y[bins == b], h["pred"][bins == b])),
                                             v53=float(roc_auc_score(y[bins == b], v["pred"][bins == b]))) for b in range(5)},
                         spearman_vs_nxa_h1=float(np.corrcoef(rank(r), rank(h["pred"]))[0, 1]))
    # independent recomputation of the reported blend grid (selection subset) and the chosen rank 0.02 blend
    r1 = np.load(R / "nxb_a_a1_day6.npz")["raw"]; rp = rank(h["pred"]); rc = rank(r1)
    grid = {f"rank{wt}": float(roc_auc_score(y[fi], ((1 - wt) * rp + wt * rc)[fi], sample_weight=s["weights"])) for wt in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)}
    out["blend_grid_sel_recomputed"] = grid; out["blend_best"] = max(grid, key=grid.get)
    bl = 0.98 * rp + 0.02 * rc
    arms = dict(h1=(h["pred"][fi], s["weights"]), blend=(bl[fi], s["weights"]))
    out["blend_rank0.02_vs_h1_selection"] = paired_many(y[fi], e[fi], arms, dict(gain=("h1", "blend")), boots)["gain"]
    arms = dict(h1=(h["pred"], w5), blend=(bl, w5))
    out["blend_rank0.02_vs_h1_full"] = paired_many(y, e, arms, dict(gain=("h1", "blend")), boots)["gain"]
    # single features on the day-6 cohort (rows of block 6 mapped exactly as nxb_a_fit.cohort_rows does)
    names = json.loads((R / "nxb_a_names.json").read_text()); meta = np.load(R / "nxb_a_meta_k6.npz"); ev = meta["events"]; e0 = ev[0]
    idx = ((e.reshape(-1, 10)[:, :1] - e0) * 10 + np.arange(10)[None, :]).ravel(); assert np.array_equal(ev[idx], e) and np.array_equal(meta["y"][idx], y)
    X = np.load(R / "nxb_a_feat_k6.npy", mmap_mode="r")
    feats = {}
    for f in ("mf_hcos", "mf_hcos_dmean", "mf_h_dmean", "cc_sum", "expo_cos", "c_hist_cnt", "u_evr", "i_ctr"):
        if f in names:
            col = np.asarray(X[idx, names.index(f)], dtype="float64"); col = np.nan_to_num(col, nan=-1e9)
            feats[f] = float(roc_auc_score(y, col))
    out["single_feature_plain_auc_day6"] = feats
    # warm16 reference columns (cached imp3_wf features, 4 columns) on the day-6 and day-5 cohorts
    for d in (6, 5):
        p = R / "features" / f"imp3_wf_val{d}.npy"
        if p.exists():
            W = np.load(p, mmap_mode="r"); vd = np.load(R / f"v53_causal_propensity_day{d}.npz")
            out[f"warm16_cols_val{d}"] = dict(shape=list(W.shape), plain_auc=[float(roc_auc_score(vd["y"], np.asarray(W[:, j]))) if W.shape[0] == len(vd["y"]) else None for j in range(W.shape[1])])
    outp.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
