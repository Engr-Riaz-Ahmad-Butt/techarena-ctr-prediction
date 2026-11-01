"""tst5: independent candidate evaluation (protocol_v4 round). No fitting; reads saved arrays only.

Usage: python tst5_candidate_eval.py NAME DAY6_NPZ DAY5_NPZ OUT_JSON [--pred KEY] [--raw KEY] [--pred39 KEY] [--boots N]

Candidate npz files must hold the V53 cohort rows (351,380 day-6 / 352,300 day-5) in V53 order with keys
`y`, `events`, the blended prediction (`pred`, default) and optionally the standalone model (`raw`) and the
recipe applied to the V39 parent (`pred_v39`).

Cohorts: day 6 full (35,138 events, project 5-bin weights), day-6 SELECTION subset (selection_day6.npz full_indices,
selection-file weights = evaluate_candidate.py convention), day 5 (35,230 events; selection-file weights == 5-bin
weights, verified). Adjusted AUC = weighted ROC AUC with those weights (history bins 0/1-4/5-9/10-19/20 to the
evaluation mix); plain = unweighted.
Baselines: nxa_h1 (artifacts/nxa_h1_day{5,6}.npz `pred`; its V39 version `pred_v39`), V53, V39.
Paired event bootstrap: 1000 resamples, seed 3 (events resampled with replacement, both arms on the same resample).
Gate (protocol_v4): >= +0.0005 adjusted vs nxa_h1 on day-6 selection AND day 5, day-6 selection paired 95% interval
above zero (full day 6 reported too). Leakage alert: standalone adjusted > 0.83 on any cohort, any blend gain vs
nxa_h1 > +0.003, or any gain > 0.01.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"
SEED = 3
H = np.load(R / "train_hist.npy", mmap_mode="r"); EH = np.load(R / "evaluation_hist.npy", mmap_mode="r")
EG = np.digitize((EH >= 0).sum(1), [0, 4, 9, 19], right=True); ETARGET = np.bincount(EG, minlength=5) / len(EG)
REF = dict(v53_d6_full=0.8171690462350286, v53_d6_sel=0.8175275258796006, v53_d5=0.8109113853236557,
           h1_d6_full=0.8180234947754393, h1_d6_sel=0.8183305782317438, h1_d5=0.8114308859896312)


def hist_bins(events):
    return np.digitize((H[events] >= 0).sum(1), [0, 4, 9, 19], right=True)


def weights5(events):  # exact copy of the project's weighted_auc weighting
    vg = hist_bins(events)
    return (ETARGET / (np.bincount(vg, minlength=5) / len(vg)))[vg]


class Plan:
    """Weighted ROC AUC under per-event multiplicities (bincount over tie groups; identical value to sklearn)."""
    def __init__(self, y, p, w, u):
        o = np.argsort(p, kind="stable"); s = p[o]; newg = np.r_[True, s[1:] != s[:-1]]; self.gid = np.cumsum(newg) - 1; self.ng = int(self.gid[-1]) + 1
        self.uo = u[o]; self.wp = (w * y)[o]; self.wn = (w * (1 - y))[o]; self.notie = self.ng == len(p)

    def value(self, c):
        mm = c[self.uo]; pw = self.wp * mm; nw = self.wn * mm
        if not self.notie: pw = np.bincount(self.gid, pw, self.ng); nw = np.bincount(self.gid, nw, self.ng)
        cn = np.cumsum(nw); return float(pw @ (cn - .5 * nw) / (pw.sum() * nw.sum()))


def paired_many(y, events, arms, pairs, boots):
    """arms: name -> (pred, weights); pairs: label -> (base_arm, cand_arm). One shared set of event resamples (seed 3, the same
    draw sequence as tst4_candidate_eval.paired, so single comparisons reproduce tst4's intervals exactly)."""
    _, u = np.unique(events, return_inverse=True); nu = int(u.max()) + 1
    used = sorted({a for pr in pairs.values() for a in pr}); plans = {a: Plan(y, arms[a][0], arms[a][1], u) for a in used}
    one = np.ones(nu); point = {a: plans[a].value(one) for a in used}
    for a in used: assert abs(point[a] - roc_auc_score(y, arms[a][0], sample_weight=arms[a][1])) < 1e-10, a
    rng = np.random.default_rng(SEED); vals = {a: np.empty(boots) for a in used}
    for i in range(boots):
        c = np.bincount(rng.integers(0, nu, nu), minlength=nu).astype(np.float64)
        for a in used: vals[a][i] = plans[a].value(c)
    out = {}
    for lab, (b, cnd) in pairs.items():
        d = vals[cnd] - vals[b]; lo, hi = np.quantile(d, [.025, .975])
        out[lab] = dict(base_arm=b, cand_arm=cnd, base=point[b], candidate=point[cnd], gain=point[cnd] - point[b], paired95=[float(lo), float(hi)], above_zero=bool(lo > 0), excludes_zero=bool(lo > 0 or hi < 0), replicates=boots, seed=SEED, events=nu)
    return out


def auc(y, p, w=None):
    return float(roc_auc_score(y, p, sample_weight=w))


def per_bin(y, p, bins):
    out = {}
    for b, name in enumerate(["0", "1-4", "5-9", "10-19", "20"]):
        m = bins == b
        out[name] = dict(rows=int(m.sum()), auc=float(roc_auc_score(y[m], p[m])) if m.any() and 0 < y[m].mean() < 1 else None)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("d6"); ap.add_argument("d5"); ap.add_argument("out")
    ap.add_argument("--pred", default="pred"); ap.add_argument("--raw", default="raw"); ap.add_argument("--pred39", default="pred_v39"); ap.add_argument("--boots", type=int, default=1000)
    a = ap.parse_args(); tic = time.monotonic(); out = dict(name=a.name, day6_file=a.d6, day5_file=a.d5, keys=dict(pred=a.pred, raw=a.raw, pred39=a.pred39))
    cohorts = {}
    for day, path in ((6, a.d6), (5, a.d5)):
        z = np.load(path); v = np.load(R / f"v53_causal_propensity_day{day}.npz"); h = np.load(R / f"nxa_h1_day{day}.npz"); s = np.load(R / f"selection_day{day}.npz")
        v39 = np.load(R / f"v39_regularized_als32_day{day}.npz")
        y = v["y"]; e = v["events"]
        chk = dict(files=list(z.files), rows=int(len(z["y"])), y_equal_v53=bool(np.array_equal(z["y"], y)), events_equal_v53=bool(np.array_equal(z["events"], e)),
                   h1_cohort_equal=bool(np.array_equal(h["y"], y) and np.array_equal(h["events"], e)), v39_cohort_equal=bool(np.array_equal(v39["y"], y) and np.array_equal(v39["events"], e)))
        assert chk["y_equal_v53"] and chk["events_equal_v53"] and chk["h1_cohort_equal"] and chk["v39_cohort_equal"], chk
        arr = dict(v53=v["pred"], v39=v39["pred"], h1=h["pred"], h1_v39=h["pred_v39"], cand=np.asarray(z[a.pred], dtype="float64"))
        if a.raw in z.files: arr["raw"] = np.asarray(z[a.raw], dtype="float64")
        if a.pred39 in z.files: arr["cand_v39"] = np.asarray(z[a.pred39], dtype="float64")
        for k, x in arr.items():
            assert x.shape == y.shape and np.isfinite(x).all(), (day, k, x.shape)
        chk["pred_identical_to_h1"] = bool(np.array_equal(arr["cand"], arr["h1"])); chk["pred_identical_to_v53"] = bool(np.array_equal(arr["cand"], arr["v53"]))
        chk["spearman_pred_vs_h1"] = float(np.corrcoef(np.argsort(np.argsort(arr["cand"])), np.argsort(np.argsort(arr["h1"])))[0, 1])
        if "raw" in arr: chk["spearman_raw_vs_v53"] = float(np.corrcoef(np.argsort(np.argsort(arr["raw"])), np.argsort(np.argsort(arr["v53"])))[0, 1])
        if day == 6:
            fi = s["full_indices"]; assert np.array_equal(s["y"], y[fi]) and np.array_equal(s["baseline"], v["pred"][fi])
            cohorts["day6_full"] = (y, e, weights5(e), arr, chk)
            cohorts["day6_selection"] = (y[fi], e[fi], s["weights"], {k: x[fi] for k, x in arr.items()}, dict(rows=int(len(fi))))
        else:
            assert np.array_equal(s["y"], y) and np.array_equal(s["baseline"], v["pred"]) and np.allclose(s["weights"], weights5(e))
            cohorts["day5"] = (y, e, s["weights"], arr, chk)
    for cname, (y, e, w, arr, chk) in cohorts.items():
        res = dict(checks=chk, rows=int(len(y)), events=int(len(np.unique(e))))
        res["point"] = {k: dict(plain=auc(y, x), adjusted=auc(y, x, w)) for k, x in arr.items()}
        one = np.ones(len(y)); arms = {k: (x, w) for k, x in arr.items()}; arms["cand_plain"] = (arr["cand"], one); arms["h1_plain"] = (arr["h1"], one)
        c39 = "cand_v39" if "cand_v39" in arr else "cand"; res["v39_key_used"] = "pred_v39" if "cand_v39" in arr else "pred"
        pairs = dict(pred_vs_h1=("h1", "cand"), pred_vs_v53=("v53", "cand"), pred39_vs_v39=("v39", c39), pred39_vs_h1_v39=("h1_v39", c39), plain_pred_vs_h1=("h1_plain", "cand_plain"))
        if "raw" in arr: pairs["raw_vs_h1"] = ("h1", "raw")
        pm = paired_many(y, e, arms, pairs, a.boots)
        res["paired_plain"] = dict(pred_vs_h1=pm.pop("plain_pred_vs_h1")); res["paired_adjusted"] = pm
        bins = hist_bins(e); res["per_history_bin_plain_auc"] = {k: per_bin(y, arr[k], bins) for k in arr if k in ("cand", "raw", "h1", "v53")}
        out[cname] = res
        print(cname, json.dumps({k: round(v_["adjusted"], 6) for k, v_ in res["point"].items()}), "gain vs h1", round(res["paired_adjusted"]["pred_vs_h1"]["gain"], 6), res["paired_adjusted"]["pred_vs_h1"]["paired95"], flush=True)
    f6, s6, d5 = out["day6_full"], out["day6_selection"], out["day5"]
    out["references_reproduced"] = dict(v53_d6_full=f6["point"]["v53"]["adjusted"], v53_d6_sel=s6["point"]["v53"]["adjusted"], v53_d5=d5["point"]["v53"]["adjusted"],
                                        h1_d6_full=f6["point"]["h1"]["adjusted"], h1_d6_sel=s6["point"]["h1"]["adjusted"], h1_d5=d5["point"]["h1"]["adjusted"])
    out["references_reproduced"]["match"] = bool(all(abs(out["references_reproduced"][k] - REF[k]) < 1e-9 for k in REF))
    g = {c: out[c]["paired_adjusted"]["pred_vs_h1"] for c in ("day6_full", "day6_selection", "day5")}
    raw_max = max((out[c]["point"]["raw"]["adjusted"] for c in g if "raw" in out[c]["point"]), default=None)
    alert_reasons = []
    if raw_max is not None and raw_max > .83: alert_reasons.append(f"standalone adjusted {raw_max:.6f} > 0.83")
    if max(x["gain"] for x in g.values()) > .003: alert_reasons.append("blend gain vs nxa_h1 > +0.003")
    allg = [out[c]["paired_adjusted"][k]["gain"] for c in g for k in out[c]["paired_adjusted"] if k != "raw_vs_h1"]
    if max(allg) > .01: alert_reasons.append("some gain > 0.01")
    acc = bool(g["day6_selection"]["gain"] >= .0005 and g["day5"]["gain"] >= .0005 and g["day6_selection"]["paired95"][0] > 0 and not alert_reasons)
    screen = bool(not acc and not alert_reasons and g["day6_selection"]["gain"] >= .0002)
    out["verdict"] = dict(baseline="nxa_h1", day6_selection_gain=g["day6_selection"]["gain"], day6_selection_paired95=g["day6_selection"]["paired95"],
                          day6_full_gain=g["day6_full"]["gain"], day6_full_paired95=g["day6_full"]["paired95"], day5_gain=g["day5"]["gain"], day5_paired95=g["day5"]["paired95"],
                          day6_full_adjusted=f6["point"]["cand"]["adjusted"], day6_selection_adjusted=s6["point"]["cand"]["adjusted"], day5_adjusted=d5["point"]["cand"]["adjusted"],
                          standalone_adjusted=dict((c, out[c]["point"]["raw"]["adjusted"]) for c in g if "raw" in out[c]["point"]),
                          exportable_v39_gain_vs_h1_v39=dict((c, out[c]["paired_adjusted"]["pred39_vs_h1_v39"]["gain"]) for c in g), v39_key_used=out["day5"]["v39_key_used"],
                          leakage_alert=bool(alert_reasons), leakage_reasons=alert_reasons, gate_days_passed=acc, screen_pass=screen,
                          status="LEAKAGE ALERT" if alert_reasons else ("PASSES BOTH DAYS (holdout pending)" if acc else ("SCREEN PASS" if screen else "REJECTED")))
    out["elapsed_seconds"] = time.monotonic() - tic
    Path(a.out).write_text(json.dumps(out, indent=2)); print(json.dumps(out["verdict"], indent=1)); print(json.dumps(out["references_reproduced"], indent=1)); print("WROTE", a.out)


if __name__ == "__main__":
    main()
