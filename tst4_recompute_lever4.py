"""tst4: no-fit re-verification of the Idea2 Step2 matched-control result.

Reads only saved artifacts under artifacts/v54_horizon/event_product_001.
Never fits a model. Recomputes every arm that can be rebuilt from saved arrays
plus saved scalar parameters (softmax temperature, Platt coefficients):
  * conditional-choice-only control (q = softmax(a * blended historical CDF))
  * boosted_basic horizon arm (saved measurement probabilities -> saved Platt)
  * untouched original scores (simplified blend on horizon, saved V53 on day6)
Arms that need a refit (logistic_*, boosted_existing/augmented, saved_v53
boosted_basic) are NOT rebuilt; their prediction arrays only exist in the
Kaggle output folder, which was not downloaded.
"""
import json, hashlib, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction")
H = B / "artifacts/v54_horizon"
R = H / "event_product_001"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "tst4_recompute_out.json"
BOOTS_REQ, SEED_REQ = 1000, 3


def ecdf(sorted_p, x):
    return (np.searchsorted(sorted_p, x, "left") + np.searchsorted(sorted_p, x, "right")) / (2 * len(sorted_p))


def platt(ab, p):
    return expit(ab[0] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + ab[1])


class Plan:
    def __init__(self, y, p, w):
        self.o = np.argsort(p, kind="stable"); self.y = y[self.o]; self.w = w[self.o]; s = p[self.o]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]

    def value(self, m):
        w = self.w * m[self.o]; p = np.add.reduceat(w * self.y, self.starts); n = np.add.reduceat(w * (1 - self.y), self.starts)
        return float(p @ (np.cumsum(n) - .5 * n) / (p.sum() * n.sum()))


def compare(y, c, p, w, users, mask, boots, seed):
    yy = y[mask].ravel(); cc = c[mask].ravel(); pp = p[mask].ravel(); ww = np.repeat(w[mask], 10)
    _, u = np.unique(users[mask], return_inverse=True); nu = u.max() + 1
    cp = Plan(yy, cc, ww); qp = Plan(yy, pp, ww); one = np.ones(len(yy)); cv = cp.value(one); pv = qp.value(one)
    assert abs(cv - roc_auc_score(yy, cc, sample_weight=ww)) < 1e-10
    assert abs(pv - roc_auc_score(yy, pp, sample_weight=ww)) < 1e-10
    rng = np.random.default_rng(seed); ds = []
    for _ in range(boots):
        count = np.bincount(rng.integers(0, nu, nu), minlength=nu); mult = np.repeat(count[u], 10); ds.append(qp.value(mult) - cp.value(mult))
    return dict(events=int(mask.sum()), users=int(nu), positive_rows=int(yy.sum()), control_auc=cv, candidate_auc=pv, delta=pv - cv,
                unweighted_control_auc=float(roc_auc_score(yy, cc)), unweighted_candidate_auc=float(roc_auc_score(yy, pp)),
                paired95=np.quantile(ds, [.025, .975]).tolist(), bootstrap_replicates=boots, seed=seed,
                effective_events=float(w[mask].sum() ** 2 / np.square(w[mask]).sum()))


def wauc(y, p, w):
    return float(roc_auc_score(y.ravel(), p.ravel(), sample_weight=np.repeat(w, 10)))


def choice_loss(a, z, target):
    return float(np.mean(logsumexp(a * z, axis=1) - a * z[np.arange(len(z)), target]))


def saved_rows(rep_case, comparison):
    return {m["group"]: m for m in rep_case["metrics"] if m["comparison"] == comparison}


def main():
    tic = time.monotonic()
    rep = json.loads((R / "kaggle_report.json").read_text()); ref = json.loads((R / "reference_comparisons.json").read_text())
    manifest = json.loads((R / "upload/manifest.json").read_text())
    result = dict(bundle_hashes={}, protected={}, horizon={}, saved_v53={}, notes=[])
    # 0. input bundle integrity
    for n, m in manifest.items():
        hsh = hashlib.sha256((R / "upload" / n).read_bytes()).hexdigest(); result["bundle_hashes"][n] = dict(expected=m["sha256"], actual=hsh, match=hsh == m["sha256"])
    # 1. protected-file hashes
    protected = json.loads((H / "protected_before.json").read_text()); mism = []
    for k, d in protected.items():
        p = B / k; ok = p.exists(); hsh = hashlib.sha256(p.read_bytes()).hexdigest() if ok else None; size = p.stat().st_size if ok else None
        match = ok and hsh == d["sha256"] and size == d["size"]; result["protected"][k] = dict(exists=ok, size_expected=d["size"], size_actual=size, sha_match=hsh == d["sha256"] if ok else False, match=bool(match))
        if not match: mism.append(k)
    result["protected_summary"] = dict(files=len(protected), mismatches=mism)

    # ---------------- horizon ----------------
    d = np.load(R / "upload/horizon_product.npz"); hz = rep["horizon"]; cal = d["cal"]; fit = ~cal
    yf = d["yfit"]; event_y = yf.sum(1); y = d["yval"]; w = d["weight"]; user = d["val_user"]; hist = d["history_length"]
    warm_sorted = np.sort(d["fit_warm"][fit].ravel()); als_sorted = np.sort(d["fit_als"][fit].ravel())
    sf = .95 * ecdf(warm_sorted, d["fit_warm"]) + .05 * ecdf(als_sorted, d["fit_als"]); sv = .95 * ecdf(warm_sorted, d["val_warm"]) + .05 * ecdf(als_sorted, d["val_als"])
    a = float(hz["inverse_softmax_temperature"]); q = softmax(a * sv, axis=1); assert np.allclose(q.sum(1), 1.)
    selected = cal & (event_y == 1); target = yf[selected].argmax(1); z = sf[selected]
    nll = choice_loss(a, z, target); nll_up = choice_loss(a * 1.001, z, target); nll_dn = choice_loss(a / 1.001, z, target)
    prior = float(event_y.mean()); choice = prior * q
    ab = hz["event_platt_parameters"]["boosted_basic"]; pe = platt(ab, d["measurement_val_probability"]); boosted = pe[:, None] * q
    assert np.allclose(boosted.sum(1), pe)
    orig = d["original_val_score"]
    arrays = dict(choice_constant=choice, boosted_basic=boosted, original=orig)
    hres = result["horizon"]
    hres["array_checks"] = dict(fit_ts_max=int(d["fit_ts"].max()), val_ts_min=int(d["val_ts"].min()), val_ts_max=int(d["val_ts"].max()), fit_ts_min=int(d["fit_ts"].min()),
                                fit_before_val=bool(d["fit_ts"].max() < d["val_ts"].min()), cal_block_chronological=bool(d["fit_ts"][fit].max() < d["fit_ts"][cal].min()),
                                fit_events=int(len(yf)), initial_fit_events=int(fit.sum()), cal_events=int(cal.sum()), val_events=int(len(y)),
                                yfit_rowsum_le1=bool((yf.sum(1) <= 1).all()), yval_rowsum_le1=bool((y.sum(1) <= 1).all()), val_positive_rows=int(y.sum()),
                                q_rows_sum_to_one=True, boosted_rows_sum_to_event_prob=True, all_finite_in_unit=bool(all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() for v in arrays.values())),
                                choice_calibration_events=int(selected.sum()), choice_calibration_events_saved=hz["choice_calibration_events"], prior_click_share=prior)
    hres["temperature_check"] = dict(saved_inverse_temperature=a, nll_at_saved=nll, nll_saved=hz["choice_calibration_nll"], nll_match=abs(nll - hz["choice_calibration_nll"]) < 1e-9,
                                     nll_up_0p1pct=nll_up, nll_down_0p1pct=nll_dn, is_local_minimum=bool(nll <= nll_up and nll <= nll_dn))
    hres["references"] = dict(original_saved_score_auc=wauc(y, orig, w), original_saved_score_auc_saved=hz["references"]["original_saved_score_auc"],
                              conditional_choice_only_auc=wauc(y, q, w), conditional_choice_only_auc_saved=hz["references"]["conditional_choice_only_auc"],
                              original_unweighted_auc=float(roc_auc_score(y.ravel(), orig.ravel())))
    # calibration metrics for boosted_basic
    cm = hz["calibration_metrics"]["boosted_basic"]
    hres["boosted_basic_calibration"] = dict(event_auc=float(roc_auc_score(y.sum(1), pe, sample_weight=w)), event_brier=float(brier_score_loss(y.sum(1), pe, sample_weight=w)),
                                             row_brier=float(brier_score_loss(y.ravel(), boosted.ravel(), sample_weight=np.repeat(w, 10))),
                                             row_logloss=float(log_loss(y.ravel(), boosted.ravel(), sample_weight=np.repeat(w, 10), labels=[0, 1])),
                                             weighted_predicted_event_click_share=float(np.average(pe, weights=w)), weighted_actual_event_click_share=float(np.average(y.sum(1), weights=w)), saved=cm)
    groups = [("overall", np.ones(len(y), bool)), ("arrival1", d["arrival"] == 0), ("arrival2", d["arrival"] == 1), ("arrival3plus", d["arrival"] >= 2),
              ("history0", hist == 0), ("history1to5", (hist >= 1) & (hist <= 5)), ("history6to20", hist >= 6)]
    saved_bvc = saved_rows(hz, "boosted_vs_choice"); saved_lvc = saved_rows(hz, "logistic_vs_choice")
    hres["boosted_vs_choice"] = {}
    for g, mask in groups:
        r_exact = compare(y, choice, boosted, w, user, mask, 200, 542201); r_req = compare(y, choice, boosted, w, user, mask, BOOTS_REQ, SEED_REQ)
        s = saved_bvc[g]
        hres["boosted_vs_choice"][g] = dict(events=r_req["events"], positive_rows=r_req["positive_rows"], ess=r_req["effective_events"],
                                           control_auc=r_req["control_auc"], control_auc_saved=s["control_auc"], candidate_auc=r_req["candidate_auc"], candidate_auc_saved=s["candidate_auc"],
                                           delta=r_req["delta"], delta_saved=s["delta"], paired95_seed542201_200=r_exact["paired95"], paired95_saved_200=s["paired95"],
                                           paired95_seed3_1000=r_req["paired95"], ess_saved=s["effective_events"],
                                           control_match=abs(r_req["control_auc"] - s["control_auc"]) < 1e-9, candidate_match=abs(r_req["candidate_auc"] - s["candidate_auc"]) < 1e-9,
                                           interval_exact_match=bool(np.allclose(r_exact["paired95"], s["paired95"], atol=1e-9)))
        # the choice-only control is shared with logistic_vs_choice: check that too
        hres["boosted_vs_choice"][g]["logistic_vs_choice_control_saved"] = saved_lvc[g]["control_auc"]
        hres["boosted_vs_choice"][g]["logistic_vs_choice_control_match"] = abs(r_req["control_auc"] - saved_lvc[g]["control_auc"]) < 1e-9
        print(json.dumps(dict(stage="horizon_group_done", group=g, elapsed=time.monotonic() - tic)), flush=True)
    # screen 20% sample
    s_ = d["sample"]; sw = d["sample_weight"]; one = np.ones(int(s_.sum()), bool)
    r_exact = compare(y[s_], choice[s_], boosted[s_], sw, user[s_], one, 400, 542201); r_req = compare(y[s_], choice[s_], boosted[s_], sw, user[s_], one, BOOTS_REQ, SEED_REQ)
    ss = [m for m in hz["screen"] if m["comparison"] == "boosted_vs_choice"][0]
    hres["boosted_vs_choice"]["screen20_overall"] = dict(events=r_req["events"], control_auc=r_req["control_auc"], control_auc_saved=ss["control_auc"], candidate_auc=r_req["candidate_auc"], candidate_auc_saved=ss["candidate_auc"],
                                                        delta=r_req["delta"], delta_saved=ss["delta"], paired95_seed542201_400=r_exact["paired95"], paired95_saved_400=ss["paired95"], paired95_seed3_1000=r_req["paired95"],
                                                        control_match=abs(r_req["control_auc"] - ss["control_auc"]) < 1e-9, candidate_match=abs(r_req["candidate_auc"] - ss["candidate_auc"]) < 1e-9,
                                                        interval_exact_match=bool(np.allclose(r_exact["paired95"], ss["paired95"], atol=1e-9)))
    # boosted_basic versus the untouched original simplified blend (reference_comparisons.json)
    rr = [m for m in ref["horizon"] if m["arm"] == "boosted_basic"][0]; mask = np.ones(len(y), bool)
    r_exact = compare(y, orig, boosted, w, user, mask, 200, 542201); r_req = compare(y, orig, boosted, w, user, mask, BOOTS_REQ, SEED_REQ)
    hres["boosted_basic_vs_original"] = dict(original_auc=r_req["control_auc"], original_auc_saved=rr["control_auc"], candidate_auc=r_req["candidate_auc"], candidate_auc_saved=rr["candidate_auc"], delta=r_req["delta"], delta_saved=rr["delta"],
                                            paired95_seed542201_200=r_exact["paired95"], paired95_saved_200=rr["paired95"], paired95_seed3_1000=r_req["paired95"],
                                            unweighted_original=r_req["unweighted_control_auc"], unweighted_original_saved=rr["unweighted_control_auc"], unweighted_candidate=r_req["unweighted_candidate_auc"], unweighted_candidate_saved=rr["unweighted_candidate_auc"],
                                            interval_exact_match=bool(np.allclose(r_exact["paired95"], rr["paired95"], atol=1e-9)))
    # choice-only versus the original simplified blend (not in the saved report; informational)
    r_req = compare(y, orig, choice, w, user, mask, BOOTS_REQ, SEED_REQ)
    hres["choice_only_vs_original"] = dict(original_auc=r_req["control_auc"], choice_auc=r_req["candidate_auc"], delta=r_req["delta"], paired95_seed3_1000=r_req["paired95"])
    # weight audit: IPF margins
    hres["weight_margins"] = dict(arrival_share=[float(w[d["arrival"] == k].sum() / w.sum()) for k in range(4)], arrival_counts=[int((d["arrival"] == k).sum()) for k in range(4)],
                                  history_share=[float(w[m].sum() / w.sum()) for _, m in groups[4:]], history_counts=[int(m.sum()) for _, m in groups[4:]], weight_mean=float(w.mean()))
    print(json.dumps(dict(stage="horizon_done", elapsed=time.monotonic() - tic)), flush=True)

    # ---------------- saved V53 day6 ----------------
    d = np.load(R / "upload/saved_v53_product.npz"); sv53 = rep["saved_v53"]; cal = d["cal"]; fit = ~cal
    yf = d["yfit"]; event_y = yf.sum(1); y = d["yval"]; w = d["weight"]; user = d["val_user"]; hist = d["history_length"]
    prior_sorted = np.sort(d["fit_score"][fit].ravel()); sf = ecdf(prior_sorted, d["fit_score"]); sv = ecdf(prior_sorted, d["val_score"])
    a = float(sv53["inverse_softmax_temperature"]); q = softmax(a * sv, axis=1); assert np.allclose(q.sum(1), 1.)
    selected = cal & (event_y == 1); target = yf[selected].argmax(1); z = sf[selected]
    nll = choice_loss(a, z, target); nll_up = choice_loss(a * 1.001, z, target); nll_dn = choice_loss(a / 1.001, z, target)
    prior = float(event_y.mean()); choice = prior * q; orig = d["original_val_score"]
    vres = result["saved_v53"]
    bins = np.where(hist == 0, 0, np.where(hist <= 5, 1, 2)); mix = np.array([.0126, .1637, .8237]); w_re = mix[bins] / (np.bincount(bins, minlength=3) / len(bins))[bins]
    vres["array_checks"] = dict(fit_ts_min=int(d["fit_ts"].min()), fit_ts_max=int(d["fit_ts"].max()), val_ts_min=int(d["val_ts"].min()), val_ts_max=int(d["val_ts"].max()),
                                fit_before_val=bool(d["fit_ts"].max() < d["val_ts"].min()), cal_block_chronological=bool(d["fit_ts"][fit].max() < d["fit_ts"][cal].min()),
                                fit_events=int(len(yf)), initial_fit_events=int(fit.sum()), cal_events=int(cal.sum()), val_events=int(len(y)), val_positive_rows=int(y.sum()),
                                yfit_rowsum_le1=bool((yf.sum(1) <= 1).all()), yval_rowsum_le1=bool((y.sum(1) <= 1).all()), val_score_is_original=bool(np.array_equal(d["val_score"], orig)),
                                weight_reproduced_from_history_mix=bool(np.allclose(w_re, w)), history_bin_counts=[int((bins == k).sum()) for k in range(3)],
                                weighted_history_share=[float(w[bins == k].sum() / w.sum()) for k in range(3)], choice_calibration_events=int(selected.sum()), choice_calibration_events_saved=sv53["choice_calibration_events"], prior_click_share=prior)
    vres["temperature_check"] = dict(saved_inverse_temperature=a, nll_at_saved=nll, nll_saved=sv53["choice_calibration_nll"], nll_match=abs(nll - sv53["choice_calibration_nll"]) < 1e-9,
                                     nll_up_0p1pct=nll_up, nll_down_0p1pct=nll_dn, is_local_minimum=bool(nll <= nll_up and nll <= nll_dn))
    vres["references"] = dict(original_v53_weighted_auc=wauc(y, orig, w), original_v53_weighted_auc_saved=sv53["references"]["original_saved_score_auc"],
                              original_v53_unweighted_auc=float(roc_auc_score(y.ravel(), orig.ravel())), original_v53_unweighted_auc_saved_ref=ref["saved_v53"][0]["unweighted_control_auc"],
                              conditional_choice_only_auc=wauc(y, q, w), conditional_choice_only_auc_saved=sv53["references"]["conditional_choice_only_auc"])
    groups = [("overall", np.ones(len(y), bool)), ("history0", hist == 0), ("history1to5", (hist >= 1) & (hist <= 5)), ("history6to20", hist >= 6)]
    saved_ctrl = saved_rows(sv53, "logistic_vs_choice")
    vres["choice_only_control_by_group"] = {}
    for g, mask in groups:
        yy = y[mask].ravel(); ww = np.repeat(w[mask], 10); v = float(roc_auc_score(yy, choice[mask].ravel(), sample_weight=ww)); o = float(roc_auc_score(yy, orig[mask].ravel(), sample_weight=ww))
        vres["choice_only_control_by_group"][g] = dict(events=int(mask.sum()), choice_only_auc=v, choice_only_auc_saved=saved_ctrl[g]["control_auc"], match=abs(v - saved_ctrl[g]["control_auc"]) < 1e-9, untouched_v53_auc=o)
    # choice-only replacement vs untouched V53 (paired, requested bootstrap)
    r_req = compare(y, orig, choice, w, user, np.ones(len(y), bool), BOOTS_REQ, SEED_REQ)
    vres["choice_only_vs_untouched_v53"] = dict(v53_auc=r_req["control_auc"], choice_only_auc=r_req["candidate_auc"], delta=r_req["delta"], paired95_seed3_1000=r_req["paired95"])
    # replacement table consistency between kaggle_report.json, reference_comparisons.json and the md report
    v53_auc = vres["references"]["original_v53_weighted_auc"]; table = []
    for rr in ref["saved_v53"]:
        cand_saved = [m for m in sv53["metrics"] if m["group"] == "overall" and m["comparison"] in ("logistic_vs_choice", "boosted_vs_choice", "logistic_beyond_existing", "boosted_beyond_existing")]
        mapping = dict(logistic_basic="logistic_vs_choice", boosted_basic="boosted_vs_choice", logistic_augmented="logistic_beyond_existing", boosted_augmented="boosted_beyond_existing")
        kr = [m for m in cand_saved if m["comparison"] == mapping[rr["arm"]]][0]
        table.append(dict(arm=rr["arm"], candidate_auc_reference=rr["candidate_auc"], candidate_auc_kaggle_report=kr["candidate_auc"], consistent=abs(rr["candidate_auc"] - kr["candidate_auc"]) < 1e-12,
                          untouched_v53_recomputed=v53_auc, untouched_v53_reference=rr["control_auc"], delta_vs_recomputed_v53=rr["candidate_auc"] - v53_auc, delta_reference=rr["delta"], paired95_reference=rr["paired95"],
                          below_v53=rr["candidate_auc"] < v53_auc, interval_excludes_zero=rr["paired95"][1] < 0))
    vres["replacement_table"] = table
    print(json.dumps(dict(stage="saved_v53_done", elapsed=time.monotonic() - tic)), flush=True)
    result["elapsed_seconds"] = time.monotonic() - tic
    result["notes"].append("Arms requiring a refit (logistic_basic/existing/augmented, boosted_existing/augmented, saved_v53 boosted_basic) were not rebuilt: their prediction arrays (frozen_predictions.npz) exist only in the Kaggle output and are not present locally.")
    OUT.write_text(json.dumps(result, indent=2)); print("WROTE", OUT)


if __name__ == "__main__":
    main()
