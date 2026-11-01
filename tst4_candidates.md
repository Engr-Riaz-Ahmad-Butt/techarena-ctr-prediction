# tst4 candidate verdicts (tester, protocol_v3 round)

Metric: adjusted AUC = project `weighted_auc` (5 history bins re-weighted to the evaluation history mix; V53 = 0.810911 day 5, 0.817169 full day 6, 0.817528 day-6 selection subset). Gate: >= +0.0005 adjusted on both days vs V53 with the day-6 paired event-bootstrap 95% interval (1000 resamples, seed 3) excluding zero; +0.0002 on both days = screen pass only. Any single gain > 0.01 = leakage until disproved. Fresh holdout scored only after both days pass. Evaluator: `tst4_candidate_eval.py`.

## v70_session_timing — REJECTED (2026-10-29 19:00)

Queue line: `v70_session_timing | 0.817191 | 0.810925 | V53 head + 12 session/candidate-recency timing columns | v70_session_timing.py | artifacts/v70_session_timing_day6.npz, artifacts/v70_session_timing_day5.npz`.
What it is: V53's exact head protocol (190 features + 12 strictly-past timing columns, logistic C=0.01, 0.75/0.25 rank blend, day-5 two-fold user cross-fit, then a full day-5 fit scored on day 6). Parent/recipe: V53 only; no V39 recipe and no evaluation-row predictions were supplied, so it is not exportable as it stands.

Recomputed by the tester (`tst4_candidate_eval.py`, output `tst4_v70_session_timing_eval.json`; V53 reference reproduced exactly: 0.817169046 full day 6, 0.817527526 day-6 selection, 0.810911385 day 5):

| Cohort | Metric | V53 | v70 | Gain | Paired event-bootstrap 95% (1000, seed 3) |
|---|---|---:|---:|---:|---|
| Full day 6 (35,138 events) | adjusted (project 5-bin code) | 0.817169046 | 0.817191275 | +0.000022229 | +0.000011912 to +0.000032659 |
| Full day 6 | plain | 0.825423332 | 0.825447099 | +0.000023767 | +0.000013803 to +0.000033403 |
| Day-6 selection subset (24,422 events) | adjusted (selection-file weights) | 0.817527526 | 0.817551127 | +0.000023601 | +0.000011708 to +0.000036483 |
| Day-6 selection subset | adjusted (3-bin mix 1.26/16.37/82.37) | 0.820037962 | 0.820061065 | +0.000023103 | +0.000010736 to +0.000035837 |
| Day-6 selection subset | plain | 0.825526709 | 0.825549295 | +0.000022586 | +0.000011255 to +0.000034434 |
| Day 5 (35,230 events) | adjusted (project 5-bin code) | 0.810911385 | 0.810925008 | +0.000013623 | +0.000001276 to +0.000026083 |
| Day 5 | plain | 0.818091384 | 0.818107301 | +0.000015917 | +0.000005113 to +0.000027340 |

The improver's numbers (0.817191 / 0.810925, +0.000022 / +0.000014) are confirmed to the digit. The gain is real but about 22 times too small for the +0.0005 gate and 9 times too small for the +0.0002 screen on day 6. Fresh holdout not scored (both-day gate not passed). Not a V39 blend, so no V39 comparison applies.

Leakage audit (code read + array checks): day-6 rows all have timestamps >= 1737849600 and day-5 rows >= 1737763200 (cutoff = start of the scored day); V53's label statistics use only rows with timestamp < cutoff (`t[t[:,2]<cut]`, asserted in `v53_causal_propensity.features`); the 12 new columns use only history click timestamps (asserted strictly before each event) and the event time, never labels; the head for day 6 is fitted on day-5 labels only; the day-5 score is user-cross-fitted (the project's standard caveat: not a chronological fold); settings are the frozen V53 settings (C=0.01, blend 0.25, seed 4801 partition, session gap 4 h), no search was run (`frozen.json` created with open('x'), one run in the log); frozen source/input hashes all still match. No single-step gain anywhere near 0.01. Verdict: clean, but far below the gate.

**Decision: REJECTED.** V53 unchanged; cumulative accepted gain this round 0.

## nxa_expo — NOT ACCEPTED (screen pass only; day 5 below the +0.0005 gate) (2026-10-29 21:15)

Queue line: exposure-policy expert (TruncatedSVD k=64, seed 4403, of the IDF-log user x item IMPRESSION matrix, no labels, impressions < cutoff); expert = mean rank of typ16_all, typ32_all, cos32; recipe `pred = rank(parent) + 0.015 * zc` (zc = expert minus its event mean, divided by its std over the scored set); parent V53 on days 5/6, V39 on days 5/6 (`pred_v39`) and on evaluation (`nxa_expo_eval_pred_v39.npy`). Scripts `nxa_exposure2.py --day 5/6/7`, `nxa_expo_final.py`.

Recomputed by the tester (`tst4_candidate_eval.py` -> `tst4_nxa_expo_eval.json`; paired event bootstrap 1000 resamples, seed 3; V53 references reproduced exactly):

| Day / cohort | Parent | Metric | Parent AUC | Candidate AUC | Gain | Paired 95% interval | Gate +0.0005 |
|---|---|---|---:|---:|---:|---|---|
| Day 6 full (35,138 events) | V53 | adjusted (project 5-bin) | 0.817169046 | 0.817810339 | **+0.000641292** | +0.000435517 to +0.000837082 | pass |
| Day 6 selection subset (24,422) | V53 | adjusted (selection weights, evaluate_candidate.py) | 0.817527526 | 0.818108905 | **+0.000581380** | +0.000358771 to +0.000830806 | pass |
| Day 6 full | V53 | plain | 0.825423332 | 0.825994665 | +0.000571333 | +0.000377978 to +0.000760449 | |
| Day 6 full | V39 | adjusted (project 5-bin) | 0.816661488 | 0.817283250 | +0.000621762 | +0.000415888 to +0.000817727 | pass |
| Day 6 selection subset | V39 | adjusted (selection weights) | 0.816993547 | 0.817561269 | +0.000567722 | +0.000345461 to +0.000817677 | pass |
| Day 5 (35,230 events) | V53 | adjusted (project 5-bin) | 0.810911385 | 0.811397720 | **+0.000486334** | +0.000299846 to +0.000685123 | **FAIL (by 0.000014)** |
| Day 5 | V53 | plain | 0.818091384 | 0.818538545 | +0.000447161 | +0.000266360 to +0.000631093 | |
| Day 5 | V39 | adjusted (project 5-bin) | 0.810526600 | 0.811000981 | **+0.000474381** | +0.000287698 to +0.000676830 | **FAIL** |

Improver's figures confirmed to the digit (+0.000641 / +0.000486; V39 +0.000622 / +0.000474). No gain anywhere near 0.01. Fresh holdout NOT scored (requires a both-days pass).

Leakage / protocol audit (`tst4_audit_nxa_expo.py` -> `tst4_nxa_expo_audit.json`):
- One code path for all three prediction sets (`features_for(day)`, then `expert` + `recipe`): cutoffs 1737763200 / 1737849600 / 1737936000. All feature files were written after the last edit of `nxa_exposure2.py` (20:32): day 6 20:33, day 5 20:37, evaluation 20:44.
- Every scored row is at or after its cutoff: day 5 1737763200 to 1737849540; day 6 1737849600 to 1737935940; evaluation 1737936000 to 1738540560.
- Impressions used = exactly the training rows strictly before the cutoff (latest 1737763140 / 1737849540 / 1737935940). The 310 rows sitting exactly at the day-5 and day-6 cutoffs are excluded. Independent replay of the label-free `n_expo` feature matches to float32 precision (max difference 4.7e-7).
- No labels in the features: the code slices `t[:n*10, :3]` (user, item, time). The single `[:, 5]` read is inside an equality assert. Labels enter `nxa_expo_final.py` only in the metric calls.
- Seeds fixed: SVD random_state 4403. Rank ties are averaged, so the output is deterministic.
- Expert and recipe recomputed with my own code reproduce the saved `raw`, `pred` and `pred_v39` arrays on days 5 and 6, and the evaluation `raw` and `pred_v39`, with a maximum difference of 0.0.
- The expert is nearly uncorrelated with V53 (Spearman 0.087), so the signal is genuinely new and not a re-encoding of V53.
- Caveat 1, transductive normalisation: the expert ranks and the zc std are computed over the whole scored set. This is label-free and matches the legacy V39 export convention. The std is stable across sets (0.2348 / 0.2410 / 0.2413).
- Caveat 2, selection cohort: the recipe was selected on FULL day 6, which contains the 10,716 fresh-holdout events (107,160 rows). The holdout is therefore no longer independent for this recipe.
- Caveat 3, selection trail: the within-event-z grid that produced beta 0.015 is not saved as an artifact. The day-5 combo screen (20:38) predates the final recipe (20:44). "Chosen on day 6 only" is plausible but cannot be verified from files.

**Decision: NOT ACCEPTED (screen pass only).** Day 6 clears the gate on both the full day and the selection subset, against both V53 and V39, with intervals excluding zero. Day 5, on the same frozen recipe, reaches +0.000486 against V53 and +0.000474 against V39, below the +0.0005 point gate. The 0.000014 shortfall is far inside the bootstrap noise, so the true gain may lie on either side of the gate. Even so, the gate is a point rule, and re-tuning on day 5 to cross it would itself break protocol. Given the multiple comparisons on day 6, day 5 is the more honest estimate of the gain. V53 is unchanged and no submission was built. This is the strongest candidate of the round: real, clean and new.

User override 2026-10-29 21:23 (relayed by the coordinator): build exploratory submission file. The verdict above stays NOT ACCEPTED under the gate. Built `submission_v40_candidate.csv` (next unused N; not renamed to a final version, the user decides after uploading) = rank(rank(V39 eval) + 0.015 * zc), from `artifacts/nxa_expo_eval_pred_v39.npy`. That array matches my independent recomputation from `v39_final_pred.npy` + `nxa_expo_eval.npy` exactly (max difference 0.0). check_submission.py passed: 529,390 rows, 0 missing/unknown/duplicate/invalid. Row order = evaluation.csv = submission_v39.csv; no NaN; all values in [0,1]. SHA-256 59304dc065349462bdf0df94a51e8a9d3c92de883f3b988ea647456415785738. Spearman vs submission_v39.csv 0.998725; top candidate changed in 3,522 of 52,939 evaluation events (6.65%). Local gain vs V39: adjusted +0.000622 (day 6 full) / +0.000568 (day 6 selection) / +0.000474 (day 5). Build record: `tst4_nxa_expo_submission_build.json`.

## nxa_h1 — ACCEPTED (2026-10-29 22:15)

What it is: an exportable logistic head over two inputs, logit(rank(parent)) and E3z (the label-free `nxa_expo` exposure expert, centred within each event and divided by its std). StandardScaler + LogisticRegression C=0.01 with evaluation-mix 5-bin sample weights. Fitting:
- day-6 predictions: fitted on the full day-5 cohort;
- day-5 predictions: 2-fold user cross-fit, split seed 9901;
- evaluation predictions: fitted on the full day-6 cohort with the V39 parent.

Scripts: `nxa_exposure2.py`, `nxa_h1.py`, `nxa_h1_final.py`. Tester files: `tst4_nxa_h1_eval.json`, `tst4_nxa_h1_audit.json`, `tst4_nxa_h1_holdout.json`, `tst4_nxa_h1_submission_build.json`.

Recomputed by the tester (paired event bootstrap, 1000 resamples, seed 3; V53 references reproduced exactly):

| Day / cohort | Compared with | Metric | Base AUC | nxa_h1 AUC | Gain | Paired 95% interval |
|---|---|---|---:|---:|---:|---|
| Day 6 full (35,138 events) | V53 | adjusted | 0.817169046 | 0.818023495 | **+0.000854449** | +0.000530082 to +0.001155628 |
| Day 6 selection subset (24,422) | V53 | adjusted (selection weights) | 0.817527526 | 0.818330578 | **+0.000803052** | +0.000460512 to +0.001169183 |
| Day 5 (35,230) | V53 | adjusted | 0.810911385 | 0.811430886 | **+0.000519501** | +0.000221723 to +0.000839161 |
| Day 6 full | V39 (head on V39) | adjusted | 0.816661488 | 0.817480929 | +0.000819441 | +0.000502452 to +0.001121083 |
| Day 5 | V39 (head on V39) | adjusted | 0.810526600 | 0.811017168 | +0.000490568 | +0.000192854 to +0.000807381 |
| Day 6 full | nxa_expo on V53 (stack increment) | adjusted | 0.817810339 | 0.818023495 | +0.000213156 | +0.000067321 to +0.000364168 |
| Day 5 | nxa_expo on V53 (stack increment) | adjusted | 0.811397720 | 0.811430886 | +0.000033166 | -0.000116531 to +0.000196758 |
| **Fresh holdout (10,716 events)** | V53 | adjusted | 0.816375401 | 0.817349127 | **+0.000973726** | +0.000416935 to +0.001568410 |
| Fresh holdout | V39 | adjusted | 0.815930483 | 0.816858196 | +0.000927713 | +0.000367613 to +0.001506985 |

Plain AUC gains vs V53: +0.000763 (day 6), +0.000504 (day 5), +0.000811 (holdout). The improver's figures were confirmed to the digit. There is no gain anywhere near 0.01.

Gate: at least +0.0005 adjusted on both days vs V53 (day 6 full and selection subset, and day 5): PASS. Day-6 interval excludes zero: PASS. Fresh holdout non-negative: PASS (+0.00097, interval above zero). The holdout was scored only after both days passed and after I confirmed the head was selected on the day-6 selection subset.

Leakage and protocol audit, with no refit (`tst4_audit_nxa_h1.py`):
- One code path for all three sets (`design()` / `fit_predict()`), with cutoffs 1737763200 / 1737849600 / 1737936000.
- Every scored row is at or after its cutoff.
- The day-6 head's labels end at 1737849540 (cutoff 1737849600). The evaluation head's labels end at 1737935940 (cutoff 1737936000).
- Day-6 predictions are exactly affine in the two claimed features (deviation 8.9e-16) using the reported coefficients, with day-5 standardisation. So the head was fitted on day-5 data only.
- Day-5 predictions are exactly affine inside each seed-9901 user fold, with different coefficients per fold, and not affine across folds (residual 0.057). So there are genuinely two cross-fitted models with disjoint users.
- Evaluation predictions are reproduced exactly (max difference 0.0) from the reported coefficients, with day-6/V39 standardisation applied to the day-7-cutoff features and `v39_final_pred.npy`.
- Seeds are fixed (SVD 4403, fold split 9901; lbfgs is deterministic).
- Selection: the grid (`nxa_h1_selection.json`, 21:36:43) was scored on the day-6 SELECTION subset only. The recipe was frozen (`nxa_h1_frozen.json`, 21:37:02, with source hashes asserted) before any day-5 output (21:37:48), and those hashes still match. F1 was chosen over F2/F3 because the extra gain of F2/F3 equals what the no-exposure recalibration control S0 gives alone. That is a sound choice.

Caveats (none is disqualifying, all recorded):
1. Day 5 uses same-day labels of other users through the user cross-fit. This is the V53 convention, so the comparison is like-for-like, but it is not a chronological fold.
2. On the exportable V39 parent, day 5 is +0.000491, 0.000009 under +0.0005. The formal gate is defined against V53, which passes.
3. On day 5 the head adds only +0.000033 over the fixed `nxa_expo` blend, and that increment is not significant. The head was designed after `nxa_expo` missed day 5 by 0.000014, so its narrow day-5 pass partly reflects a second attempt at the same signal. The fresh holdout (+0.00097, interval above zero) is the independent evidence that the signal is real.
4. The E3 expert's columns were chosen during `nxa_expo` development on FULL day 6, which includes the holdout events. The choice was among five near-identical combinations (within 0.00003 of each other), so the bias this introduces into the holdout check is tiny next to the +0.00097 gain. Even so, the holdout is not fully pristine for the expert design.

**Decision: ACCEPTED.** Built `submission_v41_candidate.csv` = `artifacts/nxa_h1_eval_pred_v39.npy`, which is already rank-transformed and was independently reproduced exactly. Details:
- check_submission.py passed: 529,390 rows; order = evaluation.csv = submission_v39.csv; no NaN; all values in [0,1].
- SHA-256 71ada3f0e16240526fb61ce4846d29dd93bd4a9369001d0795e21902de618756.
- vs submission_v39.csv: Spearman 0.995749; top candidate changed in 3,223 of 52,939 events (6.09%).
- vs submission_v40_candidate.csv: Spearman 0.998781; top candidate differs in 1,290 events (2.44%).

Honest expected gain over V39: +0.0005 to +0.0009 local adjusted (day 5 +0.00049, day 6 +0.00082, holdout +0.00093). V39 has never been uploaded, and the mapping from local to official scores is not established.

## Phase change 2026-10-29 22:42 (relayed by the coordinator)

- New baseline: nxa_h1. Each candidate is reported against nxa_h1, V53 and V39, with intervals.
- ACCEPT requires all of: at least +0.0005 adjusted vs nxa_h1 on day 6 full, the day-6 selection subset and day 5; day-6 intervals above zero; selection made on the day-6 selection subset only; and a non-negative result vs nxa_h1 on the fresh holdout. The holdout has already been scored once, for nxa_h1, so its independence is now reduced.
- LEAKAGE ALERT: any component gain above +0.01, or any single-step blend gain above +0.003. Clearing an alert requires a 1,000-row feature replay from raw arrays using pre-cutoff data only, plus confirmation that each day block's exposure matrix excludes rows at or after that block's cutoff.
- Target: day-6 full adjusted AUC 0.8200. nxa_h1 currently stands at 0.818023.

