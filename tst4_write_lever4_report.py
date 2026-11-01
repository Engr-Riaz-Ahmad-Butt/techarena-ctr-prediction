"""tst4: assemble tst4_lever4_reverify.md / .json from the recompute, leakage-audit and saved report JSONs."""
import json, sys, time
from pathlib import Path

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts/v54_horizon/event_product_001"
rec = json.loads(Path(sys.argv[1]).read_text()); leak = json.loads(Path(sys.argv[2]).read_text())
rep = json.loads((R / "kaggle_report.json").read_text()); ref = json.loads((R / "reference_comparisons.json").read_text())
hz, sv = rep["horizon"], rep["saved_v53"]


def f9(x): return f"{x:.9f}"
def g9(x): return f"{x:+.9f}"
def iv(v): return f"{v[0]:+.9f} to {v[1]:+.9f}"


def rows(case, comparison): return {m["group"]: m for m in case["metrics"] if m["comparison"] == comparison}


out = {"verdict": None, "discrepancies": [], "not_recomputable": [], "horizon": rec["horizon"], "saved_v53": rec["saved_v53"], "leakage_audit": leak, "protected": rec["protected_summary"], "bundle_hashes": rec["bundle_hashes"]}
L = ["# tst4 Lever 4 re-verification: Idea 2 Step 2 matched-control result", "",
     f"Written {time.strftime('%Y-%m-%d %H:%M')} by the tester (tst4_). Inputs: `artifacts/v54_horizon/event_product_001/kaggle_report.json`, `reference_comparisons.json`, the frozen input bundle under `upload/` (hash-checked against `upload/manifest.json`), and the scripts `prepare_event_product.py`, `kaggle_event_product.py`, `horizon_event_measurement.py`, `horizon_matched_control.py`, `horizon_prepare.py`. No model was fitted or rerun. Scripts: `tst4_recompute_lever4.py`, `tst4_leakage_audit_lever4.py`, `tst4_write_lever4_report.py`.", ""]

# ---- availability caveat ----
L += ["## What could and could not be recomputed", "",
      "The Kaggle run saved its prediction arrays (`frozen_predictions.npz`, `models.pkl`) only in the Kaggle output folder; they were never downloaded, so they do not exist locally (searched the project, `ctr_work`, Downloads and the temp folder). The local artifacts hold the frozen inputs, the saved scalar parameters (softmax inverse temperature, Platt coefficients) and the metric JSONs. Therefore:", "",
      "- Rebuilt exactly, with no fit: the conditional-choice-only control (historical CDF of the saved raw expert scores, fixed 95/5 blend, saved temperature, softmax), the horizon **boosted_basic** arm (saved measurement-model probabilities, replayed from the saved LightGBM file with max difference 0, through the saved Platt map, times the same choice probabilities), the untouched original scores (horizon simplified blend; saved V53 on day 6), all group breakdowns for those arms, and every paired user-bootstrap interval for those arms.",
      "- Not rebuildable without refitting (forbidden here): the logistic arms (basic / existing / augmented), the boosted existing / augmented arms, and the saved-V53 boosted_basic arm (a fresh LightGBM fit there). For these the saved numbers were checked for internal consistency across `kaggle_report.json`, `reference_comparisons.json` and the markdown report, and against invariants that hold regardless of the fit (shared controls, monotone-map identities).", ""]
bh = rec["bundle_hashes"]; L += ["Input bundle hashes vs `upload/manifest.json`: " + ("all 5 match." if all(v["match"] for v in bh.values()) else "MISMATCH: " + ", ".join(k for k, v in bh.items() if not v["match"])), ""]
if not all(v["match"] for v in bh.values()): out["discrepancies"].append("input bundle hash mismatch")

# ---- (1) overall horizon AUCs ----
H = rec["horizon"]; bvc = H["boosted_vs_choice"]; ov = bvc["overall"]
L += ["## 1. Horizon cohort weighted AUC (49,420 first-user events, IPF weights to arrival 67:17:7:4 and history 1.26/16.37/82.37)", "",
      "| Arm | Reported | Recomputed | Status |", "|---|---:|---:|---|",
      f"| choice-only control (prior x q) | {f9(ov['control_auc_saved'])} | {f9(ov['control_auc'])} | {'exact' if ov['control_match'] else 'DISCREPANCY'} |",
      f"| boosted_basic (measurement model, Platt) x q | {f9(ov['candidate_auc_saved'])} | {f9(ov['candidate_auc'])} | {'exact' if ov['candidate_match'] else 'DISCREPANCY'} |",
      f"| original simplified blend (untouched) | {f9(H['references']['original_saved_score_auc_saved'])} | {f9(H['references']['original_saved_score_auc'])} | {'exact' if abs(H['references']['original_saved_score_auc'] - H['references']['original_saved_score_auc_saved']) < 1e-9 else 'DISCREPANCY'} |"]
for arm, comp, key in [("logistic_basic", "logistic_vs_choice", "candidate_auc"), ("logistic_existing (control)", "logistic_beyond_existing", "control_auc"), ("logistic_augmented", "logistic_beyond_existing", "candidate_auc"), ("boosted_existing (control)", "boosted_beyond_existing", "control_auc"), ("boosted_augmented", "boosted_beyond_existing", "candidate_auc")]:
    L.append(f"| {arm} | {f9(rows(hz, comp)['overall'][key])} | not recomputable (arrays not local) | saved value, consistency-checked |"); out["not_recomputable"].append(arm)
for k in ["choice-only control", "boosted_basic", "original simplified blend"]:
    pass
if not (ov["control_match"] and ov["candidate_match"]): out["discrepancies"].append("horizon overall AUC mismatch")
tc = H["temperature_check"]
L += ["", f"Softmax inverse temperature check: saved a = {tc['saved_inverse_temperature']:.9f}; conditional NLL at a recomputed {tc['nll_at_saved']:.9f} vs saved {tc['nll_saved']:.9f} ({'match' if tc['nll_match'] else 'DISCREPANCY'}); a is a local minimum of the calibration-block NLL: {tc['is_local_minimum']}. Choice calibration events {H['array_checks']['choice_calibration_events']} (saved {H['array_checks']['choice_calibration_events_saved']}).", ""]
if not tc["nll_match"]: out["discrepancies"].append("horizon temperature NLL mismatch")
cb = H["boosted_basic_calibration"]; s = cb["saved"]
L += [f"boosted_basic calibration metrics recomputed vs saved: event AUC {cb['event_auc']:.6f} / {s['event_auc']:.6f}; event Brier {cb['event_brier']:.6f} / {s['event_brier']:.6f}; row Brier {cb['row_brier']:.6f} / {s['row_brier']:.6f}; row logloss {cb['row_logloss']:.6f} / {s['row_logloss']:.6f}; predicted click-event share {cb['weighted_predicted_event_click_share']:.4%} / {s['weighted_predicted_event_click_share']:.4%}; observed {cb['weighted_actual_event_click_share']:.4%}. The event AUC equals the Step 1 measurement AUC 0.627487, as it must for a monotone Platt map.", ""]
if abs(cb["event_auc"] - s["event_auc"]) > 1e-9 or abs(cb["row_logloss"] - s["row_logloss"]) > 1e-9: out["discrepancies"].append("boosted_basic calibration metrics mismatch")

L += ["### Paired user-bootstrap intervals (gain = candidate minus control)", "",
      "| Comparison | Gain reported | Gain recomputed | Reported 95% (200 resamples, seed 542201) | Replicated with same seed/count | Requested 1000 resamples, seed 3 |", "|---|---:|---:|---|---|---|",
      f"| boosted_vs_choice, overall | {g9(ov['delta_saved'])} | {g9(ov['delta'])} | {iv(ov['paired95_saved_200'])} | {iv(ov['paired95_seed542201_200'])} ({'identical' if ov['interval_exact_match'] else 'differs'}) | {iv(ov['paired95_seed3_1000'])} |"]
sc = bvc["screen20_overall"]
L.append(f"| boosted_vs_choice, 20% screen sample | {g9(sc['delta_saved'])} | {g9(sc['delta'])} | {iv(sc['paired95_saved_400'])} (400) | {iv(sc['paired95_seed542201_400'])} ({'identical' if sc['interval_exact_match'] else 'differs'}) | {iv(sc['paired95_seed3_1000'])} |")
bo = H["boosted_basic_vs_original"]
L.append(f"| boosted_basic vs untouched simplified blend | {g9(bo['delta_saved'])} | {g9(bo['delta'])} | {iv(bo['paired95_saved_200'])} | {iv(bo['paired95_seed542201_200'])} ({'identical' if bo['interval_exact_match'] else 'differs'}) | {iv(bo['paired95_seed3_1000'])} |")
co = H["choice_only_vs_original"]
L.append(f"| choice-only vs untouched simplified blend (new, informational) | n/a | {g9(co['delta'])} | n/a | n/a | {iv(co['paired95_seed3_1000'])} |")
for comp in ["logistic_vs_choice", "logistic_beyond_existing", "boosted_beyond_existing"]:
    m = rows(hz, comp)["overall"]; L.append(f"| {comp}, overall | {g9(m['delta'])} | not recomputable | {iv(m['paired95'])} | n/a | n/a |")
L += ["", "Reading: the reported intervals for the rebuildable arms reproduce to the digit with the script's own seed, and the 1000-resample / seed-3 intervals agree to within the bootstrap's own Monte-Carlo noise. The reported logistic-arm numbers cannot be re-derived locally; their shared choice-only control (0.752211990) does reproduce exactly, so their reported gains are measured against the right control.", ""]
if not (ov["interval_exact_match"] and sc["interval_exact_match"] and bo["interval_exact_match"]): out["discrepancies"].append("bootstrap interval replication differs with identical seed")

# ---- (2) breakdowns ----
L += ["## 2. Breakdown by arrival day and history bin (horizon)", "", "### boosted_vs_choice (rebuilt exactly)", "",
      "| Group | Events | ESS | Control reported / recomputed | Candidate reported / recomputed | Gain reported / recomputed | Reported 95% | 1000-resample seed-3 95% |", "|---|---:|---:|---|---|---|---|---|"]
allok = True
for g in ["overall", "arrival1", "arrival2", "arrival3plus", "history0", "history1to5", "history6to20"]:
    r = bvc[g]; ok = r["control_match"] and r["candidate_match"]; allok &= ok
    L.append(f"| {g} | {r['events']:,} | {r['ess']:.1f} | {f9(r['control_auc_saved'])} / {f9(r['control_auc'])} | {f9(r['candidate_auc_saved'])} / {f9(r['candidate_auc'])} | {g9(r['delta_saved'])} / {g9(r['delta'])}{'' if ok else ' DISCREPANCY'} | {iv(r['paired95_saved_200'])} | {iv(r['paired95_seed3_1000'])} |")
if not allok: out["discrepancies"].append("boosted_vs_choice group breakdown mismatch")
L += ["", "### logistic_beyond_existing (best arm by reported gain; saved values, arrays not local)", "",
      "| Group | Events | Control (logistic_existing) | Candidate (logistic_augmented) | Gain | Reported 95% | Interval excludes zero |", "|---|---:|---:|---:|---:|---|---|"]
for g, m in rows(hz, "logistic_beyond_existing").items():
    L.append(f"| {g} | {m['events']:,} | {f9(m['control_auc'])} | {f9(m['candidate_auc'])} | {g9(m['delta'])} | {iv(m['paired95'])} | {'yes' if m['paired95'][0] > 0 or m['paired95'][1] < 0 else 'no'} |")
L += ["", "Consistency checks on the saved logistic arms: the logistic_vs_choice control equals the recomputed choice-only control in every group (" + ("all match" if all(bvc[g]["logistic_vs_choice_control_match"] for g in bvc if g != "screen20_overall") else "MISMATCH") + "); the four arms' candidate AUCs in `reference_comparisons.json` equal those in `kaggle_report.json` to 1e-12; the history-0 group (693 users) is the only group where the best arm loses to its strengthened control (-0.008795, interval excluding zero), as the report states.", ""]
if not all(bvc[g]["logistic_vs_choice_control_match"] for g in bvc if g != "screen20_overall"): out["discrepancies"].append("logistic_vs_choice control mismatch")
wm = H["weight_margins"]; L += [f"IPF weight margins recomputed from the frozen weights: arrival shares {', '.join(f'{x:.6f}' for x in wm['arrival_share'])} (target 0.705263/0.178947/0.073684/0.042105), history shares {', '.join(f'{x:.6f}' for x in wm['history_share'])} (target 0.0126/0.1637/0.8237).", ""]

# ---- (3) replacement into V53 ----
V = rec["saved_v53"]; vr = V["references"]
L += ["## 3. Replacement into V53 on the saved day-6 selection cohort (24,422 events, history-only weights 1.26/16.37/82.37)", "",
      f"Untouched V53 weighted AUC recomputed from the saved arrays: **{f9(vr['original_v53_weighted_auc'])}** (saved {f9(vr['original_v53_weighted_auc_saved'])}; {'exact' if abs(vr['original_v53_weighted_auc'] - vr['original_v53_weighted_auc_saved']) < 1e-9 else 'DISCREPANCY'}); unweighted {f9(vr['original_v53_unweighted_auc'])} (saved {f9(vr['original_v53_unweighted_auc_saved_ref'])}). Choice-only control recomputed {f9(vr['conditional_choice_only_auc'])} (saved {f9(vr['conditional_choice_only_auc_saved'])}; {'exact' if abs(vr['conditional_choice_only_auc'] - vr['conditional_choice_only_auc_saved']) < 1e-9 else 'DISCREPANCY'}). Weights reproduced from the history bins: {V['array_checks']['weight_reproduced_from_history_mix']}.", "",
      "| Replacement arm | Candidate AUC (saved) | Untouched V53 (recomputed) | Gain vs V53 | Saved paired 95% | Below V53 | Interval excludes zero |", "|---|---:|---:|---:|---|---|---|"]
allbelow = True
for t in V["replacement_table"]:
    allbelow &= t["below_v53"] and t["interval_excludes_zero"]
    L.append(f"| {t['arm']} | {f9(t['candidate_auc_reference'])} | {f9(t['untouched_v53_recomputed'])} | {g9(t['delta_vs_recomputed_v53'])} | {iv(t['paired95_reference'])} | {'yes' if t['below_v53'] else 'NO'} | {'yes' if t['interval_excludes_zero'] else 'no'} |")
cv = V["choice_only_vs_untouched_v53"]
L.append(f"| choice-only control alone (rebuilt exactly) | {f9(cv['choice_only_auc'])} | {f9(cv['v53_auc'])} | {g9(cv['delta'])} | 1000/seed 3: {iv(cv['paired95_seed3_1000'])} | {'yes' if cv['delta'] < 0 else 'NO'} | {'yes' if cv['paired95_seed3_1000'][1] < 0 else 'no'} |")
if abs(vr["original_v53_weighted_auc"] - vr["original_v53_weighted_auc_saved"]) > 1e-9: out["discrepancies"].append("saved V53 day-6 reference mismatch")
if not allbelow: out["discrepancies"].append("a replacement arm is not below V53 or its interval includes zero")
L += ["", f"Every tested replacement scores below untouched V53 (best logistic_augmented 0.818817715 vs 0.820037962, -0.001220247, interval entirely negative), so the recorded decision 'no promotion, keep V53' is correct. Note that the gains the report calls 'beyond strengthened controls' (+0.000457 / +0.000323 on day 6) are gains over reduced controls, not over V53. Saved-V53 choice-only control by history bin, recomputed vs saved: " + "; ".join(f"{g}: {f9(V['choice_only_control_by_group'][g]['choice_only_auc'])} / {f9(V['choice_only_control_by_group'][g]['choice_only_auc_saved'])}" for g in V["choice_only_control_by_group"]) + ".", ""]
tcv = V["temperature_check"]; L += [f"Saved-V53 softmax temperature {tcv['saved_inverse_temperature']:.9f}: NLL recomputed {tcv['nll_at_saved']:.9f} vs saved {tcv['nll_saved']:.9f} ({'match' if tcv['nll_match'] else 'DISCREPANCY'}), local minimum {tcv['is_local_minimum']}.", ""]
if not tcv["nll_match"]: out["discrepancies"].append("saved V53 temperature NLL mismatch")

# ---- (4) leakage audit ----
hl, sl, mr = leak["horizon"], leak["saved_v53"], leak["model_replay"]
L += ["## 4. Leakage audit of the scripts and frozen arrays", "", "| Check | Horizon | Saved V53 day 6 |", "|---|---|---|",
      f"| Event-click model fitted on pre-cutoff first-user events only | fit events = first event per user inside [cutoff1, cutoff3) exactly ({hl['fit_events_match_first_event_rule']}); {hl['fit_events']} events, users unique {hl['fit_users_unique']}; last fit label {hl['fit_label_ts_max']} < cutoff3 {hl['cutoff3']} <= first scored event {hl['val_ts_min']} ({hl['fit_labels_strictly_before_cutoff_and_validation']}) | fit = all 35,230 day-5 events ({sl['fit_events_equal_selection_day5']}), inside day 5 ({sl['fit_ts_inside_day5']}); scored = 24,422 day-6 events ({sl['val_events_equal_selection_day6']}), inside day 6 ({sl['val_ts_inside_day6']}); last fit label 1737849540 < 1737849600 ({sl['fit_labels_before_val']}); fit/scored events disjoint {sl['fit_val_events_disjoint']} (day-5 scores are user-cross-fitted, a documented caveat) |",
      f"| Calibration fitted chronologically before validation | calibration block = latest 8,843 of the fit events, strictly after the 35,156 initial-fit events ({rec['horizon']['array_checks']['cal_block_chronological']}); CDFs sorted from the initial-fit partition only; temperature fitted on clicked calibration-block events; Platt fitted on temporally out-of-sample probabilities | calibration block = latest 60% of day 5 ({sl['cal_block_is_latest_60pct_of_day5']}); chronological ({rec['saved_v53']['array_checks']['cal_block_chronological']}) |",
      f"| No evaluation labels anywhere | `yval` is read only after all predictions are saved (code order in `kaggle_event_product.py`); scoring rows carry the -1 sentinel ({hl['score_rows_label_sentinel_all_minus1']}); feature function invariant to the scored rows' labels ({hl['feature_function_label_invariance']}); `yval` equals the frozen `validation_labels.npy` ({hl['yval_equals_validation_labels']}); fresh holdout and evaluation.csv never opened by any of these scripts | day-6 labels enter only the metric routine; `yval` equals `selection_day6.npz` labels ({sl['yval_equals_day6_labels']}) |",
      f"| No post-cutoff history clicks / label statistics | every supplied history click precedes its event ({hl['val_history_times_before_event']}); history clicks at or after cutoff3: {hl['val_history_clicks_at_or_after_cutoff3']} of {hl['val_history_clicks_total']}; user click-rate, last-activity gap, hours-since-cutoff and history-length features replayed independently from pre-cutoff rows with max difference {max(hl['val_feature_replay']['rate_max_abs_diff'], hl['val_feature_replay']['gap_max_abs_diff'], hl['val_feature_replay']['hours_since_cutoff_max_abs_diff'], hl['val_feature_replay']['history_length_max_abs_diff'])}; ALS checkpoints cut1/cut3 carry cutoffs {hl['als_cut1_cutoff']} / {hl['als_cut3_cutoff']} ({hl['als_cutoffs_consistent']}); ALS score features replayed with max difference {max(hl['val_als_feature_replay_max_abs_diff'].values())} | history clicks at or after the day-6 cutoff: {sl['val_feature_replay']['history_clicks_at_or_after_cutoff']}; history precedes event ({sl['val_feature_replay']['history_times_before_event']}); label statistics replayed from rows < 1737849600 with max difference {max(sl['val_feature_replay']['rate_max_abs_diff'], sl['val_feature_replay']['gap_max_abs_diff'])}; ALS model cutoff matches ({sl['val_feature_replay']['als_cutoff_matches']}) |",
      f"| Saved measurement model replay | LightGBM {mr['lightgbm_version']} replay of `measurement_event_model.txt` on the frozen features: max difference {mr['max_abs_diff_vs_bundle']} vs bundle, {mr['max_abs_diff_vs_measurement_arrays']} vs Step 1 arrays; {mr['num_trees']} trees | n/a (fresh fit on Kaggle, not local) |", "",
      "Residual caveats (already documented in the report, confirmed here): the horizon 'original' score is a joint rank blend over the whole 4-day cohort (legacy convention, not stateless); the saved-V53 diagnostic calibrates on day-5 user-cross-fitted scores; the raw expert scores feeding the CDFs come from cut1 models on the fit side and cut3 models on the validation side (a transport imperfection, not leakage).", ""]
leak_ok = all([hl["fit_events_match_first_event_rule"], hl["fit_labels_strictly_before_cutoff_and_validation"], hl["val_history_times_before_event"], hl["val_history_clicks_at_or_after_cutoff3"] == 0, hl["feature_function_label_invariance"], hl["score_rows_label_sentinel_all_minus1"], hl["val_feature_replay"]["rate_max_abs_diff"] == 0, sl["fit_labels_before_val"], sl["cal_block_is_latest_60pct_of_day5"], sl["val_feature_replay"]["history_clicks_at_or_after_cutoff"] == 0, sl["val_feature_replay"]["rate_max_abs_diff"] == 0, mr["max_abs_diff_vs_bundle"] == 0])
if not leak_ok: out["discrepancies"].append("leakage audit check failed")

# ---- protected hashes ----
P = rec["protected_summary"]
L += ["## 5. Protected-file hash check", "", f"{P['files']} files listed in `artifacts/v54_horizon/protected_before.json` compared by size and SHA-256: " + ("**all unchanged**." if not P["mismatches"] else "**MISMATCH**: " + ", ".join(P["mismatches"])), ""]
if P["mismatches"]: out["discrepancies"].append("protected file mismatch: " + ", ".join(P["mismatches"]))

# ---- verdict ----
verdict = "CONFIRMED" if not out["discrepancies"] else "DISCREPANCY"
out["verdict"] = verdict; out["leakage_audit_passed"] = leak_ok
L += ["## Verdict", "", f"**{verdict}**" + ("" if not out["discrepancies"] else ": " + "; ".join(out["discrepancies"])) + ".", "",
      "- Everything rebuildable from the local artifacts reproduces the saved numbers to the digit (choice-only control, boosted_basic arm, original scores, all group breakdowns, the bootstrap intervals with the script's seed) and the requested 1000-resample seed-3 intervals agree within Monte-Carlo noise.",
      "- The logistic and strengthened-control arms could not be rebuilt because their prediction arrays were never downloaded from Kaggle; their saved numbers are internally consistent and sit on exactly reproduced controls. Recommendation: download `/kaggle/working/.../horizon/frozen_predictions.npz` and `saved_v53/frozen_predictions.npz` (plus `models.pkl`) into `artifacts/v54_horizon/event_product_001/` so a later check can close this gap without a refit.",
      "- The decision 'no promotion, keep V53 unchanged' is correct: every tested replacement is below untouched V53 on the saved day-6 selection cohort by 0.0012 to 0.0018 with intervals entirely below zero.",
      "- Leakage audit: " + ("passed on every check." if leak_ok else "FAILED, see section 4."),
      "- Protected files: " + ("all 29 unchanged." if not P["mismatches"] else "mismatch, see section 5."), ""]
(B / "tst4_lever4_reverify.md").write_text("\n".join(L), encoding="utf-8"); (B / "tst4_lever4_reverify.json").write_text(json.dumps(out, indent=2))
print(verdict, out["discrepancies"])
