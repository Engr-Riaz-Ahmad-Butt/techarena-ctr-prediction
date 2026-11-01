# tst4 Lever 4 re-verification: Idea 2 Step 2 matched-control result

Written 2026-10-29 18:55 by the tester (tst4_). Inputs: `artifacts/v54_horizon/event_product_001/kaggle_report.json`, `reference_comparisons.json`, the frozen input bundle under `upload/` (hash-checked against `upload/manifest.json`), and the scripts `prepare_event_product.py`, `kaggle_event_product.py`, `horizon_event_measurement.py`, `horizon_matched_control.py`, `horizon_prepare.py`. No model was fitted or rerun. Scripts: `tst4_recompute_lever4.py`, `tst4_leakage_audit_lever4.py`, `tst4_write_lever4_report.py`.

## What could and could not be recomputed

The Kaggle run saved its prediction arrays (`frozen_predictions.npz`, `models.pkl`) only in the Kaggle output folder; they were never downloaded, so they do not exist locally (searched the project, `ctr_work`, Downloads and the temp folder). The local artifacts hold the frozen inputs, the saved scalar parameters (softmax inverse temperature, Platt coefficients) and the metric JSONs. Therefore:

- Rebuilt exactly, with no fit: the conditional-choice-only control (historical CDF of the saved raw expert scores, fixed 95/5 blend, saved temperature, softmax), the horizon **boosted_basic** arm (saved measurement-model probabilities, replayed from the saved LightGBM file with max difference 0, through the saved Platt map, times the same choice probabilities), the untouched original scores (horizon simplified blend; saved V53 on day 6), all group breakdowns for those arms, and every paired user-bootstrap interval for those arms.
- Not rebuildable without refitting (forbidden here): the logistic arms (basic / existing / augmented), the boosted existing / augmented arms, and the saved-V53 boosted_basic arm (a fresh LightGBM fit there). For these the saved numbers were checked for internal consistency across `kaggle_report.json`, `reference_comparisons.json` and the markdown report, and against invariants that hold regardless of the fit (shared controls, monotone-map identities).

Input bundle hashes vs `upload/manifest.json`: all 5 match.

## 1. Horizon cohort weighted AUC (49,420 first-user events, IPF weights to arrival 67:17:7:4 and history 1.26/16.37/82.37)

| Arm | Reported | Recomputed | Status |
|---|---:|---:|---|
| choice-only control (prior x q) | 0.752211990 | 0.752211990 | exact |
| boosted_basic (measurement model, Platt) x q | 0.755036395 | 0.755036395 | exact |
| original simplified blend (untouched) | 0.755280328 | 0.755280328 | exact |
| logistic_basic | 0.755326063 | not recomputable (arrays not local) | saved value, consistency-checked |
| logistic_existing (control) | 0.754645383 | not recomputable (arrays not local) | saved value, consistency-checked |
| logistic_augmented | 0.755554457 | not recomputable (arrays not local) | saved value, consistency-checked |
| boosted_existing (control) | 0.754461954 | not recomputable (arrays not local) | saved value, consistency-checked |
| boosted_augmented | 0.755173224 | not recomputable (arrays not local) | saved value, consistency-checked |

Softmax inverse temperature check: saved a = 2.546210699; conditional NLL at a recomputed 2.130460765 vs saved 2.130460765 (match); a is a local minimum of the calibration-block NLL: True. Choice calibration events 5723 (saved 5723).

boosted_basic calibration metrics recomputed vs saved: event AUC 0.627487 / 0.627487; event Brier 0.222183 / 0.222183; row Brier 0.055352 / 0.055352; row logloss 0.209801 / 0.209801; predicted click-event share 63.9527% / 63.9527%; observed 63.0959%. The event AUC equals the Step 1 measurement AUC 0.627487, as it must for a monotone Platt map.

### Paired user-bootstrap intervals (gain = candidate minus control)

| Comparison | Gain reported | Gain recomputed | Reported 95% (200 resamples, seed 542201) | Replicated with same seed/count | Requested 1000 resamples, seed 3 |
|---|---:|---:|---|---|---|
| boosted_vs_choice, overall | +0.002824405 | +0.002824405 | +0.002348530 to +0.003304004 | +0.002348530 to +0.003304004 (identical) | +0.002370880 to +0.003267762 |
| boosted_vs_choice, 20% screen sample | +0.001909970 | +0.001909970 | +0.000911773 to +0.003024905 (400) | +0.000911773 to +0.003024905 (identical) | +0.000849920 to +0.002904355 |
| boosted_basic vs untouched simplified blend | -0.000243934 | -0.000243934 | -0.000792384 to +0.000275471 | -0.000792384 to +0.000275471 (identical) | -0.000760885 to +0.000327181 |
| choice-only vs untouched simplified blend (new, informational) | n/a | -0.003068338 | n/a | n/a | -0.003536277 to -0.002549177 |
| logistic_vs_choice, overall | +0.003114073 | not recomputable | +0.002626556 to +0.003533568 | n/a | n/a |
| logistic_beyond_existing, overall | +0.000909074 | not recomputable | +0.000676759 to +0.001152789 | n/a | n/a |
| boosted_beyond_existing, overall | +0.000711270 | not recomputable | +0.000510908 to +0.000892471 | n/a | n/a |

Reading: the reported intervals for the rebuildable arms reproduce to the digit with the script's own seed, and the 1000-resample / seed-3 intervals agree to within the bootstrap's own Monte-Carlo noise. The reported logistic-arm numbers cannot be re-derived locally; their shared choice-only control (0.752211990) does reproduce exactly, so their reported gains are measured against the right control.

## 2. Breakdown by arrival day and history bin (horizon)

### boosted_vs_choice (rebuilt exactly)

| Group | Events | ESS | Control reported / recomputed | Candidate reported / recomputed | Gain reported / recomputed | Reported 95% | 1000-resample seed-3 95% |
|---|---:|---:|---|---|---|---|---|
| overall | 49,420 | 48907.6 | 0.752211990 / 0.752211990 | 0.755036395 / 0.755036395 | +0.002824405 / +0.002824405 | +0.002348530 to +0.003304004 | +0.002370880 to +0.003267762 |
| arrival1 | 34,912 | 34784.1 | 0.762409388 / 0.762409388 | 0.765146088 / 0.765146088 | +0.002736700 / +0.002736700 | +0.002191326 to +0.003293373 | +0.002189468 to +0.003312494 |
| arrival2 | 8,988 | 8920.5 | 0.733118355 / 0.733118355 | 0.736548003 / 0.736548003 | +0.003429648 / +0.003429648 | +0.002421334 to +0.004498345 | +0.002478311 to +0.004438560 |
| arrival3plus | 5,520 | 5242.3 | 0.715385210 / 0.715385210 | 0.717534908 / 0.717534908 | +0.002149697 / +0.002149697 | +0.001014307 to +0.003303223 | +0.001042437 to +0.003381222 |
| history0 | 693 | 675.7 | 0.517799192 / 0.517799192 | 0.527017844 / 0.527017844 | +0.009218653 / +0.009218653 | -0.002661097 to +0.021731712 | -0.002345907 to +0.021370470 |
| history1to5 | 9,265 | 9103.8 | 0.672092508 / 0.672092508 | 0.675985684 / 0.675985684 | +0.003893176 / +0.003893176 | +0.002642618 to +0.005154998 | +0.002749568 to +0.004988640 |
| history6to20 | 39,462 | 39290.8 | 0.766389430 / 0.766389430 | 0.769097332 / 0.769097332 | +0.002707902 / +0.002707902 | +0.002234327 to +0.003196666 | +0.002204997 to +0.003198537 |

### logistic_beyond_existing (best arm by reported gain; saved values, arrays not local)

| Group | Events | Control (logistic_existing) | Candidate (logistic_augmented) | Gain | Reported 95% | Interval excludes zero |
|---|---:|---:|---:|---:|---|---|
| overall | 49,420 | 0.754645383 | 0.755554457 | +0.000909074 | +0.000676759 to +0.001152789 | yes |
| arrival1 | 34,912 | 0.765057502 | 0.765806585 | +0.000749083 | +0.000515080 to +0.001048386 | yes |
| arrival2 | 8,988 | 0.735657568 | 0.736727368 | +0.001069800 | +0.000584260 to +0.001546031 | yes |
| arrival3plus | 5,520 | 0.717040157 | 0.717390224 | +0.000350068 | -0.000451247 to +0.001002453 | no |
| history0 | 693 | 0.529528270 | 0.520733160 | -0.008795110 | -0.016587870 to -0.001133254 | yes |
| history1to5 | 9,265 | 0.674564280 | 0.676068267 | +0.001503987 | +0.000724100 to +0.002299576 | yes |
| history6to20 | 39,462 | 0.768883160 | 0.769724359 | +0.000841199 | +0.000640406 to +0.001080766 | yes |

Consistency checks on the saved logistic arms: the logistic_vs_choice control equals the recomputed choice-only control in every group (all match); the four arms' candidate AUCs in `reference_comparisons.json` equal those in `kaggle_report.json` to 1e-12; the history-0 group (693 users) is the only group where the best arm loses to its strengthened control (-0.008795, interval excluding zero), as the report states.

IPF weight margins recomputed from the frozen weights: arrival shares 0.705263, 0.178947, 0.073684, 0.042105 (target 0.705263/0.178947/0.073684/0.042105), history shares 0.012600, 0.163700, 0.823700 (target 0.0126/0.1637/0.8237).

## 3. Replacement into V53 on the saved day-6 selection cohort (24,422 events, history-only weights 1.26/16.37/82.37)

Untouched V53 weighted AUC recomputed from the saved arrays: **0.820037962** (saved 0.820037962; exact); unweighted 0.825526709 (saved 0.825526709). Choice-only control recomputed 0.814831678 (saved 0.814831678; exact). Weights reproduced from the history bins: True.

| Replacement arm | Candidate AUC (saved) | Untouched V53 (recomputed) | Gain vs V53 | Saved paired 95% | Below V53 | Interval excludes zero |
|---|---:|---:|---:|---|---|---|
| logistic_basic | 0.818455959 | 0.820037962 | -0.001582003 | -0.001953029 to -0.001192313 | yes | yes |
| boosted_basic | 0.818278967 | 0.820037962 | -0.001758995 | -0.002129430 to -0.001431430 | yes | yes |
| logistic_augmented | 0.818817715 | 0.820037962 | -0.001220247 | -0.001552856 to -0.000877860 | yes | yes |
| boosted_augmented | 0.818633973 | 0.820037962 | -0.001403989 | -0.001716730 to -0.001103424 | yes | yes |
| choice-only control alone (rebuilt exactly) | 0.814831678 | 0.820037962 | -0.005206285 | 1000/seed 3: -0.005708026 to -0.004669734 | yes | yes |

Every tested replacement scores below untouched V53 (best logistic_augmented 0.818817715 vs 0.820037962, -0.001220247, interval entirely negative), so the recorded decision 'no promotion, keep V53' is correct. Note that the gains the report calls 'beyond strengthened controls' (+0.000457 / +0.000323 on day 6) are gains over reduced controls, not over V53. Saved-V53 choice-only control by history bin, recomputed vs saved: overall: 0.814831678 / 0.814831678; history0: 0.460487385 / 0.460487385; history1to5: 0.753785389 / 0.753785389; history6to20: 0.826472407 / 0.826472407.

Saved-V53 softmax temperature 4.801510288: NLL recomputed 1.729214323 vs saved 1.729214323 (match), local minimum True.

## 4. Leakage audit of the scripts and frozen arrays

| Check | Horizon | Saved V53 day 6 |
|---|---|---|
| Event-click model fitted on pre-cutoff first-user events only | fit events = first event per user inside [cutoff1, cutoff3) exactly (True); 43999 events, users unique True; last fit label 1737590340 < cutoff3 1737590400 <= first scored event 1737590400 (True) | fit = all 35,230 day-5 events (True), inside day 5 (True); scored = 24,422 day-6 events (True), inside day 6 (True); last fit label 1737849540 < 1737849600 (True); fit/scored events disjoint True (day-5 scores are user-cross-fitted, a documented caveat) |
| Calibration fitted chronologically before validation | calibration block = latest 8,843 of the fit events, strictly after the 35,156 initial-fit events (True); CDFs sorted from the initial-fit partition only; temperature fitted on clicked calibration-block events; Platt fitted on temporally out-of-sample probabilities | calibration block = latest 60% of day 5 (True); chronological (True) |
| No evaluation labels anywhere | `yval` is read only after all predictions are saved (code order in `kaggle_event_product.py`); scoring rows carry the -1 sentinel (True); feature function invariant to the scored rows' labels (True); `yval` equals the frozen `validation_labels.npy` (True); fresh holdout and evaluation.csv never opened by any of these scripts | day-6 labels enter only the metric routine; `yval` equals `selection_day6.npz` labels (True) |
| No post-cutoff history clicks / label statistics | every supplied history click precedes its event (True); history clicks at or after cutoff3: 0 of 624569; user click-rate, last-activity gap, hours-since-cutoff and history-length features replayed independently from pre-cutoff rows with max difference 0.0; ALS checkpoints cut1/cut3 carry cutoffs 1737417600 / 1737590400 (True); ALS score features replayed with max difference 0.0 | history clicks at or after the day-6 cutoff: 0; history precedes event (True); label statistics replayed from rows < 1737849600 with max difference 0.0; ALS model cutoff matches (True) |
| Saved measurement model replay | LightGBM 4.7.0 replay of `measurement_event_model.txt` on the frozen features: max difference 0.0 vs bundle, 0.0 vs Step 1 arrays; 120 trees | n/a (fresh fit on Kaggle, not local) |

Residual caveats (already documented in the report, confirmed here): the horizon 'original' score is a joint rank blend over the whole 4-day cohort (legacy convention, not stateless); the saved-V53 diagnostic calibrates on day-5 user-cross-fitted scores; the raw expert scores feeding the CDFs come from cut1 models on the fit side and cut3 models on the validation side (a transport imperfection, not leakage).

## 5. Protected-file hash check

29 files listed in `artifacts/v54_horizon/protected_before.json` compared by size and SHA-256: **all unchanged**.

## Verdict

**CONFIRMED**.

- Everything rebuildable from the local artifacts reproduces the saved numbers to the digit (choice-only control, boosted_basic arm, original scores, all group breakdowns, the bootstrap intervals with the script's seed) and the requested 1000-resample seed-3 intervals agree within Monte-Carlo noise.
- The logistic and strengthened-control arms could not be rebuilt because their prediction arrays were never downloaded from Kaggle; their saved numbers are internally consistent and sit on exactly reproduced controls. Recommendation: download `/kaggle/working/.../horizon/frozen_predictions.npz` and `saved_v53/frozen_predictions.npz` (plus `models.pkl`) into `artifacts/v54_horizon/event_product_001/` so a later check can close this gap without a refit.
- The decision 'no promotion, keep V53 unchanged' is correct: every tested replacement is below untouched V53 on the saved day-6 selection cohort by 0.0012 to 0.0018 with intervals entirely below zero.
- Leakage audit: passed on every check.
- Protected files: all 29 unchanged.
