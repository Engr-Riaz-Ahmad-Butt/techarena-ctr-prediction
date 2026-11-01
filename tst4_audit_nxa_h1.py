"""tst4: no-fit audit of nxa_h1 (logistic head over [logit(rank(parent)), E3z]).

A StandardScaler + LogisticRegression decision function is affine in the raw features, so without fitting anything:
 (a) day 6: pred6 - ((X6 - mean5) / std5) @ coef_reported must be constant  -> head standardised on DAY-5 features
     (i.e. fitted on day 5) and uses exactly these two inputs with the reported coefficients;
 (b) day 5: within each seed-9901 user fold, pred5 must be exactly affine in X5, with DIFFERENT coefficients per fold
     (two cross-fitted models), and not one affine map across both folds;
 (c) evaluation: rank(((Xe - mean6_v39) / std6_v39) @ coef_eval_reported) must equal the saved evaluation array
     -> evaluation head standardised on the day-6 cohort with the V39 parent, applied to day-7-cutoff features;
 (d) V39-parent day-5/6 predictions are affine in their own [logit(rank(V39)), E3z] (least squares on predictions only,
     no labels).
Plus: label-fit windows vs cutoffs, scored rows vs cutoffs, file provenance.
"""
import json, sys, time, hashlib, os
from pathlib import Path
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
START, DAY = 1737331200, 86400; COLS = ("typ16_all", "typ32_all", "cos32")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "tst4_nxa_h1_audit.json"


def rank(x): x = np.asarray(x, np.float64); return rankdata(x, method="average") / (len(x) + 1)


def zc(x): c = x - np.repeat(x.reshape(-1, 10).mean(1), 10); return c / c.std()


def X(parent, f): return np.column_stack([logit(np.clip(rank(parent), 1e-6, 1 - 1e-6)), zc(np.mean([rank(f[c]) for c in COLS], axis=0))])


def affine_resid(pred, Xm):
    A = np.column_stack([np.ones(len(pred)), Xm]); coef, *_ = np.linalg.lstsq(A, pred, rcond=None); r = pred - A @ coef
    return float(np.abs(r).max()), coef.tolist()


def main():
    tic = time.monotonic(); out = {}
    rep = json.loads((R / "nxa_h1_report.json").read_text()); c6 = np.array(rep["coef_day6_head_v53"]); ce = np.array(rep["eval"]["coef"])
    t = np.load(R / "train.npy", mmap_mode="r")
    f5 = np.load(R / "nxa_exposure2_day5_features.npz"); f6 = np.load(R / "nxa_exposure2_day6_features.npz"); f7 = np.load(R / "nxa_exposure2_day7_features.npz")
    v5 = np.load(R / "v53_causal_propensity_day5.npz"); v6 = np.load(R / "v53_causal_propensity_day6.npz")
    q5 = np.load(R / "v39_regularized_als32_day5.npz")["pred"]; q6 = np.load(R / "v39_regularized_als32_day6.npz")["pred"]
    h5 = np.load(R / "nxa_h1_day5.npz"); h6 = np.load(R / "nxa_h1_day6.npz"); he = np.load(R / "nxa_h1_eval_pred_v39.npy"); pe = np.load(R / "v39_final_pred.npy")
    out["files_consistent"] = dict(day5_y_events=bool(np.array_equal(h5["y"], v5["y"]) and np.array_equal(h5["events"], v5["events"])), day6_y_events=bool(np.array_equal(h6["y"], v6["y"]) and np.array_equal(h6["events"], v6["events"])),
                                   baseline_is_v53=bool(np.array_equal(h5["baseline"], v5["pred"]) and np.array_equal(h6["baseline"], v6["pred"])),
                                   raw_is_e3_expert=bool(np.allclose(h6["raw"], np.mean([rank(f6[c]) for c in COLS], axis=0), atol=0, rtol=0)))
    # (a) day-6 head on V53: standardised on day-5 design
    X5 = X(v5["pred"], f5); X6 = X(v6["pred"], f6); m5, s5 = X5.mean(0), X5.std(0); m6, s6 = X6.mean(0), X6.std(0)
    d6 = ((X6 - m5) / s5) @ c6; diff = h6["pred"] - d6; out["a_day6_head_v53"] = dict(intercept_implied=float(diff.mean()), max_dev_from_constant=float(np.abs(diff - diff.mean()).max()),
                                                                                        alt_if_standardised_on_day6_max_dev=float(np.abs((h6["pred"] - ((X6 - m6) / s6) @ c6) - (h6["pred"] - ((X6 - m6) / s6) @ c6).mean()).max()))
    # (b) day-5 cross-fit on V53
    users = np.asarray(t[h5["events"].astype(np.int64) * 10, 0]); uniq = np.unique(users); first = set(np.random.default_rng(9901).permutation(uniq)[: len(uniq) // 2].tolist()); fold = np.array([u in first for u in users])
    ra, ca = affine_resid(h5["pred"][fold], X5[fold]); rb, cb = affine_resid(h5["pred"][~fold], X5[~fold]); rj, cj = affine_resid(h5["pred"], X5)
    out["b_day5_crossfit_v53"] = dict(fold_rows=[int(fold.sum()), int((~fold).sum())], fold_users_disjoint=bool(len(np.intersect1d(users[fold], users[~fold])) == 0),
                                      fold_a_affine_max_resid=ra, fold_b_affine_max_resid=rb, joint_affine_max_resid=rj, fold_a_coef=ca, fold_b_coef=cb)
    # (c) evaluation head: standardised on day-6 V39 design, applied to day-7 features with v39_final_pred
    X6q = X(q6, f6); m6q, s6q = X6q.mean(0), X6q.std(0); Xe = X(pe, f7); de = ((Xe - m6q) / s6q) @ ce
    out["c_eval_head_v39"] = dict(max_abs_diff_rank_vs_saved=float(np.abs(rank(de) - he).max()), saved_min=float(he.min()), saved_max=float(he.max()),
                                  alt_if_standardised_on_day5_max_abs_diff=float(np.abs(rank(((Xe - X(q5, f5).mean(0)) / X(q5, f5).std(0)) @ ce) - he).max()))
    # (d) V39-parent validation predictions affine in their own features
    r6q, c6q = affine_resid(h6["pred_v39"], X6q); X5q = X(q5, f5); r5a, _ = affine_resid(h5["pred_v39"][fold], X5q[fold]); r5b, _ = affine_resid(h5["pred_v39"][~fold], X5q[~fold])
    out["d_v39_parent"] = dict(day6_affine_max_resid=r6q, day6_coef=c6q, day5_fold_a_resid=r5a, day5_fold_b_resid=r5b)
    # label windows vs cutoffs
    ts5 = np.asarray(t[v5["events"].astype(np.int64) * 10, 2]); ts6 = np.asarray(t[v6["events"].astype(np.int64) * 10, 2])  # row = 10 * event
    ev = np.load(R / "evaluation.npy", mmap_mode="r"); tse = np.asarray(ev[:, 2])
    out["time"] = dict(day6_head_label_rows_ts_max=int(ts5.max()), day6_cutoff=START + 6 * DAY, day6_head_labels_before_cutoff=bool(ts5.max() < START + 6 * DAY),
                       eval_head_label_rows_ts_max=int(ts6.max()), eval_cutoff=START + 7 * DAY, eval_head_labels_before_cutoff=bool(ts6.max() < START + 7 * DAY),
                       day5_scored_min=int(ts5.min()), day6_scored_min=int(ts6.min()), eval_scored_min=int(tse.min()),
                       scored_at_or_after_cutoff=bool(ts5.min() >= START + 5 * DAY and ts6.min() >= START + 6 * DAY and tse.min() >= START + 7 * DAY),
                       day5_note="day-5 predictions use same-day labels of OTHER users (2-fold user cross-fit) - the V53 convention, not a chronological fold")
    out["provenance"] = {p: time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(B / p))) for p in ["nxa_h1.py", "nxa_h1_final.py", "artifacts/nxa_h1_selection.json", "artifacts/nxa_h1_frozen.json", "artifacts/nxa_h1_day5.npz", "artifacts/nxa_h1_day6.npz", "artifacts/nxa_h1_eval_pred_v39.npy", "artifacts/nxa_exposure2_day5_features.npz", "artifacts/nxa_exposure2_day6_features.npz", "artifacts/nxa_exposure2_day7_features.npz"]}
    out["elapsed_seconds"] = time.monotonic() - tic; OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
