"""tst4: independent candidate evaluation against V53's saved predictions. No fitting.

Usage: python tst4_candidate_eval.py NAME DAY6_NPZ DAY5_NPZ OUT_JSON

Adjusted metric = the project's own `weighted_auc` (v39_regularized_als32.py /
crossfit_calibration.py): rows re-weighted so the validation history-length
distribution (5 bins: 0, 1-4, 5-9, 10-19, 20 supplied clicks) matches the
evaluation file's. This is the code that produced V53's 0.817169 (full day 6,
351,380 rows) and 0.810911 (day 5). The 3-bin mix (0 / 1-5 / 6-20 = 1.26 /
16.37 / 82.37 percent, distribution-batch convention, V53 day-6 selection
0.820038) and the selection-file weights (evaluate_candidate.py convention,
V53 day-6 selection 0.817528) are reported as well for cross-reference.
Paired event bootstrap: 1000 resamples, seed 3, events resampled with
replacement, both arms scored on the same resample.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
BOOTS, SEED = 1000, 3
H = np.load(R / "train_hist.npy", mmap_mode="r"); EH = np.load(R / "evaluation_hist.npy", mmap_mode="r")
EG = np.digitize((EH >= 0).sum(1), [0, 4, 9, 19], right=True); ETARGET = np.bincount(EG, minlength=5) / len(EG)
MIX3 = np.array([.0126, .1637, .8237])


def weights5(events):  # exact copy of the project's weighted_auc weighting
    vg = np.digitize((H[events] >= 0).sum(1), [0, 4, 9, 19], right=True)
    return (ETARGET / (np.bincount(vg, minlength=5) / len(vg)))[vg]


def weights3(events):  # distribution-batch convention
    hl = (H[events] >= 0).sum(1); bins = np.where(hl == 0, 0, np.where(hl <= 5, 1, 2))
    return (MIX3 / (np.bincount(bins, minlength=3) / len(bins)))[bins]


class Plan:
    def __init__(self, y, p, w):
        self.o = np.argsort(p, kind="stable"); self.y = y[self.o]; self.w = w[self.o]; s = p[self.o]; self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]

    def value(self, m):
        w = self.w * m[self.o]; p = np.add.reduceat(w * self.y, self.starts); n = np.add.reduceat(w * (1 - self.y), self.starts); return float(p @ (np.cumsum(n) - .5 * n) / (p.sum() * n.sum()))


def paired(y, base, cand, w, events):
    _, u = np.unique(events, return_inverse=True); nu = u.max() + 1; bp = Plan(y, base, w); cp = Plan(y, cand, w); one = np.ones(len(y))
    b = bp.value(one); c = cp.value(one); assert abs(b - roc_auc_score(y, base, sample_weight=w)) < 1e-10 and abs(c - roc_auc_score(y, cand, sample_weight=w)) < 1e-10
    rng = np.random.default_rng(SEED); ds = []
    for _ in range(BOOTS):
        count = np.bincount(rng.integers(0, nu, nu), minlength=nu); m = count[u]; ds.append(cp.value(m) - bp.value(m))
    lo, hi = np.quantile(ds, [.025, .975]); return dict(v53=b, candidate=c, gain=c - b, paired95=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0), replicates=BOOTS, seed=SEED, events=int(nu))


def evaluate_day(day, path, out):
    z = np.load(path); v = np.load(R / f"v53_causal_propensity_day{day}.npz"); sel = np.load(R / f"selection_day{day}.npz")
    y = z["y"]; e = z["events"]; p = z["pred"]
    checks = dict(rows=int(len(y)), y_equals_v53=bool(np.array_equal(y, v["y"])), events_equal_v53=bool(np.array_equal(e, v["events"])), baseline_equals_v53_pred=bool("baseline" in z.files and np.array_equal(z["baseline"], v["pred"])),
                  pred_finite=bool(np.isfinite(p).all()), pred_identical_to_v53=bool(np.array_equal(p, v["pred"])), rows_per_event_10=bool(np.array_equal(e, np.repeat(e[::10], 10))))
    assert checks["y_equals_v53"] and checks["events_equal_v53"], "candidate cohort differs from V53 cohort"
    res = dict(checks=checks)
    full_w5 = weights5(e); res["full_day"] = dict(rows=int(len(y)), events=int(len(np.unique(e))), plain=paired(y, v["pred"], p, np.ones(len(y)), e), adjusted_5bin_project_code=paired(y, v["pred"], p, full_w5, e))
    # selection subset (day 6: 24,422 events via selection_day6 full_indices; day 5: whole day is the selection cohort)
    if "full_indices" in sel.files:
        fi = sel["full_indices"]; assert np.array_equal(sel["y"], y[fi]) and np.array_equal(sel["baseline"], v["pred"][fi]); ys, es, ps, bs = y[fi], e[fi], p[fi], v["pred"][fi]
    else:
        assert np.array_equal(sel["y"], y) and np.array_equal(sel["baseline"], v["pred"]); ys, es, ps, bs = y, e, p, v["pred"]
    res["selection_cohort"] = dict(rows=int(len(ys)), events=int(len(np.unique(es))), plain=paired(ys, bs, ps, np.ones(len(ys)), es), adjusted_selection_file_weights=paired(ys, bs, ps, sel["weights"], es),
                                   adjusted_5bin_project_code=paired(ys, bs, ps, weights5(es), es), adjusted_3bin_mix=paired(ys, bs, ps, weights3(es), es))
    # V39 parent (last exportable chain): same cohorts, same metrics
    v39 = np.load(R / f"v39_regularized_als32_day{day}.npz"); assert np.array_equal(v39["y"], y) and np.array_equal(v39["events"], e); b39 = v39["pred"]
    b39s = b39[sel["full_indices"]] if "full_indices" in sel.files else b39
    # if the candidate supplies the recipe applied to V39 ('pred_v39'), compare THAT with V39; otherwise compare 'pred' with V39
    p39 = z["pred_v39"] if "pred_v39" in z.files else p; res["v39_comparison_key"] = "pred_v39" if "pred_v39" in z.files else "pred"
    p39s = p39[sel["full_indices"]] if "full_indices" in sel.files else p39
    res["vs_v39_full_day"] = dict(plain=paired(y, b39, p39, np.ones(len(y)), e), adjusted_5bin_project_code=paired(y, b39, p39, full_w5, e))
    res["vs_v39_selection_cohort"] = dict(plain=paired(ys, b39s, p39s, np.ones(len(ys)), es), adjusted_selection_file_weights=paired(ys, b39s, p39s, sel["weights"], es))
    # optional extra baseline (e.g. nxa_expo-on-V53 for stacked candidates): argv[5]=name, argv[6]=day6 npz, argv[7]=day5 npz, key 'pred'
    if len(sys.argv) > 7:
        xb = np.load(sys.argv[6] if day == 6 else sys.argv[7]); assert np.array_equal(xb["y"], y) and np.array_equal(xb["events"], e); bx = xb["pred"]
        bxs = bx[sel["full_indices"]] if "full_indices" in sel.files else bx
        res["vs_extra_" + sys.argv[5]] = dict(full_adjusted_5bin=paired(y, bx, p, full_w5, e), selection_adjusted_selection_file_weights=paired(ys, bxs, ps, sel["weights"], es))
    out[f"day{day}"] = res; return res


def main():
    name, d6, d5, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); tic = time.monotonic(); out = dict(name=name, day6_file=str(d6), day5_file=str(d5))
    r6 = evaluate_day(6, d6, out); r5 = evaluate_day(5, d5, out)
    g6 = r6["full_day"]["adjusted_5bin_project_code"]; g5 = r5["full_day"]["adjusted_5bin_project_code"]
    out["v53_reference_reproduced"] = dict(day6_adjusted=g6["v53"], day6_expected=0.8171690462350286, day5_adjusted=g5["v53"], day5_expected=0.8109113853236557,
                                           match=bool(abs(g6["v53"] - 0.8171690462350286) < 1e-9 and abs(g5["v53"] - 0.8109113853236557) < 1e-9))
    s6 = r6["selection_cohort"]["adjusted_selection_file_weights"]  # evaluate_candidate.py gate cohort for day 6
    leak_alert = bool(max(g6["gain"], g5["gain"], s6["gain"], r6["full_day"]["plain"]["gain"], r5["full_day"]["plain"]["gain"], r6["vs_v39_full_day"]["adjusted_5bin_project_code"]["gain"], r5["vs_v39_full_day"]["adjusted_5bin_project_code"]["gain"]) > .01)
    accepted = bool(g6["gain"] >= .0005 and s6["gain"] >= .0005 and g5["gain"] >= .0005 and g6["excludes_zero"] and s6["paired95"][0] > 0 and not leak_alert)
    screen = bool(g6["gain"] >= .0002 and g5["gain"] >= .0002 and not accepted and not leak_alert)
    v6 = r6["vs_v39_full_day"]["adjusted_5bin_project_code"]; v5 = r5["vs_v39_full_day"]["adjusted_5bin_project_code"]
    out["verdict"] = dict(metric="adjusted AUC, project weighted_auc code (5 history bins to evaluation mix), full day 6 / day 5", day6_gain=g6["gain"], day5_gain=g5["gain"], day6_paired95=g6["paired95"], day5_paired95=g5["paired95"],
                          day6_selection_gain=s6["gain"], day6_selection_paired95=s6["paired95"],
                          day6_gain_vs_v39=v6["gain"], day5_gain_vs_v39=v5["gain"], day6_paired95_vs_v39=v6["paired95"], day5_paired95_vs_v39=v5["paired95"],
                          accepted=accepted, screen_pass=screen, leakage_alert_gain_over_0p01=leak_alert, status="ACCEPTED" if accepted else ("SCREEN PASS ONLY" if screen else ("LEAKAGE ALERT" if leak_alert else "NOT ACCEPTED")))
    # gate against the extra baseline when given (phase rule from 2026-10-29 22:3x: baseline = nxa_h1)
    if len(sys.argv) > 7:
        xn = "vs_extra_" + sys.argv[5]; x6 = r6[xn]["full_adjusted_5bin"]; x6s = r6[xn]["selection_adjusted_selection_file_weights"]; x5 = r5[xn]["full_adjusted_5bin"]
        alert = bool(max(x6["gain"], x6s["gain"], x5["gain"]) > .003 or leak_alert)
        acc = bool(x6["gain"] >= .0005 and x6s["gain"] >= .0005 and x5["gain"] >= .0005 and x6["paired95"][0] > 0 and x6s["paired95"][0] > 0 and not alert)
        out["verdict_vs_" + sys.argv[5]] = dict(baseline=sys.argv[5], day6_full_gain=x6["gain"], day6_full_paired95=x6["paired95"], day6_selection_gain=x6s["gain"], day6_selection_paired95=x6s["paired95"],
                                                day5_gain=x5["gain"], day5_paired95=x5["paired95"], day6_full_adjusted_candidate=g6["candidate"], reached_0p8200_day6_full=bool(g6["candidate"] >= .8200),
                                                leakage_alert_blend_gain_over_0p003=alert, accepted=acc, status="LEAKAGE ALERT" if alert else ("ACCEPTED (pending holdout)" if acc else "NOT ACCEPTED"))
        print(json.dumps(out["verdict_vs_" + sys.argv[5]], indent=1))
    out["elapsed_seconds"] = time.monotonic() - tic; outp.write_text(json.dumps(out, indent=2)); print(json.dumps(out["verdict"], indent=1)); print(json.dumps(out["v53_reference_reproduced"], indent=1)); print("WROTE", outp)


if __name__ == "__main__":
    main()
