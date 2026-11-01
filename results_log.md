# TechArena CTR results log

Validation is time-based. Keep a change only if it gains at least +0.0005 on
both selection days and passes the untouched holdout before promotion.

## 2026-10-29 — V54 error analysis and candidate-generation forensics

- **Hypothesis:** identify the V53 failure segments and determine whether the
  ten-item candidate construction contains usable time-safe signals.
- **Change:** read the 686,824 training events in chronological order. At each
  event, candidate exposure, click count, and exact slate-repeat count were
  read before that event's labels updated them. V53 predictions for days 5/6
  were grouped by history length, candidate warmth, time bucket, day type, and
  list position.
- **Runtime:** 28 seconds locally. No model, validation artifact, or
  submission was overwritten.
- **Results:** V53 ordinary AUC was 0.818091 (day 5) and 0.825423 (day 6).
  Zero-history users were only 0.56% of rows and near-random (0.514/0.522 AUC);
  1--5 history clicks were 10.3%/9.3% and weak (0.745/0.761). Day-6 time
  bucket 3 was weak (0.810). Clicked items were at the 57.3%/57.0%
  prior-exposure percentile and 57.9%/57.5% smoothed-CTR percentile. A
  history-repeat candidate had about 29.7%/30.0% click rate versus about 5.7%
  for non-repeat candidates. Exact prior candidate-list repetition was 0%.
- **Decision:** diagnostic only; no candidate was kept. Screen time-safe
  event-composition features, concentrating on short histories and time bucket
  3. Details: `artifacts/v54_v53_error_and_forensics.json`.

The prior malformed results-log bytes were preserved as `results_log.legacy.bin`.

## 2026-10-29 — Event-composition evidence review

- **Hypothesis:** relative candidate-set information could complement V53.
- **Evidence reviewed:** the prior time-safe, permutation-equivariant slate
  residual experiment (`artifacts/v58_slate_residual_day6_report.json`) and the
  new forensics. The prior experiment used whole ten-item lists and a matched
  zero-slate control.
- **Result:** the slate arm's day-6 adjusted gain over V53 was only +0.000031,
  and it was -0.000009 versus the matched control. Its confidence interval
  crossed zero, far below the +0.0005 rule.
- **Decision:** reject slate-only composition as a duplicate direction; do not
  spend a second expensive run on it. Proceed to time-safe transition features
  (last 1/2/3 history items), which are materially different from slate context.

## 2026-10-29 — Directed last-item transition screen

- **Hypothesis:** counts for "last history item -> candidate click" at lags
  1--3 provide a direct sequential signal absent from V53.
- **Change:** a deterministic 20% sample of the day-6 users (6,977 events,
  69,770 rows) was scored with transition counts built only from 588,873
  earlier events. No model was fit and validation labels were not used to build
  any count.
- **Runtime:** under one second after cached-memory loading.
- **Results:** V53 AUC 0.825433; raw maximum-transition AUC 0.509809.
  A 2% rank blend fell to 0.825295, and every larger blend was worse. Only
  0.23%, 0.19%, and 0.17% of rows had a nonzero lag-1, lag-2, or lag-3 count.
- **Decision:** reject exact directed transitions as too sparse; do not run a
  full model confirmation. Next distinct route is restricted ItemKNN/EASE-like
  similarity, whose aggregation can avoid exact-pair sparsity.

## 2026-10-29 — Experiment 1: restricted ItemKNN screen

- **Hypothesis:** popular-item co-click similarity may aggregate enough evidence to improve V53 beyond sparse exact transitions.
- **Change:** binary user-item clicks strictly before each day cutoff; symmetric co-click cosine, zero diagonal; top 5k/10k/20k items; shrinkage 1/10/100; last 5/20 history clicks; max/mean/recency aggregation; rank blends 1/2/5/10%. Average ranks preserve ties. Cached sparse matrices. ItemKNN tested; dense EASE inverse not tested.
- **Runtime:** 21.09 seconds locally on CPU; 216 configurations per day on deterministic 20% user samples (7,017 day-5 events; 4,896 day-6 events).
- **Metric:** current user brief specifies global ROC AUC; older history-adjusted gates are superseded. Adjusted scores retained as secondary diagnostics. Sample scores are not full-day scores.
- **Day 5:** baseline 0.820437259; best joint configuration 0.820436035; gain -0.000001225.
- **Day 6:** baseline 0.825858710; best joint configuration 0.825846067; gain -0.000012643.
- **Result:** every tested blend worsened sample global AUC. Even top-20k, last-20 similarity is nonzero for only 7.39%/9.71% of day-5/day-6 rows.
- **Decision:** reject tested ItemKNN family; no full-data confirmation justified, no candidate promoted. V53 cumulative gain remains zero. Holdout and evaluation files were not opened; no submission written. Pause after experiment 1 as explicitly requested. Next route after user review: repeat-item deep dive.
- **Leakage controls:** only training labels strictly before cutoff used for popularity and similarity; no evaluation statistics; no validation labels used to fit matrices; self-similarity excluded. Supplied histories follow the existing data contract (prior clicks).

## 2026-10-29 — Experiment 2: repeat-item and negative-exposure deep dive

- **Hypothesis:** repeated click recency and ignored user-item presentations add residual signal beyond V53.
- **Existing-feature review:** V53 explicitly includes candidate history click count; core/core_v2 add repeat count and recent-3 matching; core_v5 and V53 contain user/item exposure/CTR summaries; core_v6 contains exposure-adjusted latent user profiles. No explicit same-user same-item ignored-presentation count, recency, or negative/click ratio found in the inspected active feature lineage. New pair features are distinct; repeat count retained as an existing control.
- **Change:** strict pre-cutoff user-item unclicked exposure count, last unclicked gap, smoothed negative/click ratio and pair click count; smoothed user seen/new CTR and seen-click share; repeat clicks-ago, last-click gap, same-bucket history fraction, user/item repeat-click rate and user bucket/same-bucket repeat CTR. Warm-up exposures unknown, so seen/new uses observed training presentations only.
- **Protocol:** deterministic same 20% user selection; day5 two-fold user OOF, day6 trained on day5 only; 180 fixed CatBoost trees, depth4, seed5402; control/repeat/exposure/combined arms, rank blends 5/10/25%. No early stopping. Global AUC is primary; history-adjusted AUC secondary. Caches contain no evaluation data.
- **Day 5 diagnostics:** repeat AUC 0.669243; non-repeat AUC 0.811391; negative-pair exposure coverage 1.0118%. Full history-length and repeat cross-tab in artifacts/v54_repeat_diagnostics.json.
- **Day 6 diagnostics:** repeat AUC 0.668188; non-repeat AUC 0.815542; negative-pair exposure coverage 1.1785%. Full history-length and repeat cross-tab in artifacts/v54_repeat_diagnostics.json.
- **control, 5% blend:** day 5 0.820437259 -> 0.820402153 (-0.000035106); day 6 0.825858710 -> 0.825864873 (+0.000006162).
- **repeat, 5% blend:** day 5 0.820437259 -> 0.820405467 (-0.000031793); day 6 0.825858710 -> 0.825856365 (-0.000002345).
- **exposure, 5% blend:** day 5 0.820437259 -> 0.820408493 (-0.000028767); day 6 0.825858710 -> 0.825873723 (+0.000015013).
- **combined, 5% blend:** day 5 0.820437259 -> 0.820407709 (-0.000029550); day 6 0.825858710 -> 0.825855783 (-0.000002927).
- **Runtime:** feature preparation 10.95s; model screening 15.88s locally, CPU.
- **Decision:** reject tested repeat/exposure screen. No configuration gains +0.0005 on both days; best shared exposure-only 5% blend gains -0.000029/+0.000015, incremental to matched control just +0.000006/+0.000009. V53 remains best, cumulative gain zero. Strong repeat click rate does not establish useful residual predictability.
- **Audit:** independent brute-force pair counts, ratios and gaps pass; all pre-cutoff supplied-history timestamps checked; no single-step +0.01; holdout/evaluation unopened. No baseline/submission overwritten. Full confirmation is not justified. Pause after experiment 2 as requested. Dense EASE excluded by user.

## 2026-10-29 — Requested screen A

- Hypothesis: cold/short histories or repeat population mismatch might explain the missing global AUC. Descriptive segment counts only. Oracle zero-history bound includes cross-segment pairs.
- Runtime: 8.906 seconds locally on CPU.
- Full-selection zero-history oracle gain: +0.002292053/+0.002237276. Evaluation shares in artifacts/v54_abcd_A.json; no labels used.
- Details: artifacts/v54_abcd_A.json. Final audit/report: results/v54_abcd_final_report.md.

## 2026-10-29 — Requested screen B

- Hypothesis: cross-segment ordering improves with a shallow regularized LightGBM stacker or per-segment isotonic. Fit approved early40% day5; evaluate later60% and future day6.
- Runtime: 3.375 seconds locally on CPU.
- Best shared configuration: shallow_lightgbm_w0.1. day5: 0.812626070 -> 0.812646380 (+0.000020310); day6: 0.825858710 -> 0.825828799 (-0.000029911).
- Decision: reject; fails +0.0005 on both days. Full cohort/holdout not evaluated.
- Details: artifacts/v54_abcd_B.json. Final audit/report: results/v54_abcd_final_report.md.

## 2026-10-29 — Requested screen C

- Hypothesis: a repeat-only expert can improve within-repeat discrimination. Eight specified features; only repeats fit/scored; non-repeat V53 scores unchanged.
- Runtime: 1.375 seconds locally on CPU.
- Best shared configuration: repeat_only_w0.1. day5: 0.812626070 -> 0.812637741 (+0.000011671); day6: 0.825858710 -> 0.825836094 (-0.000022616).
- Decision: reject; fails +0.0005 on both days. Full cohort/holdout not evaluated.
- Details: artifacts/v54_abcd_C.json. Final audit/report: results/v54_abcd_final_report.md.

## 2026-10-29 — Requested screen D

- Hypothesis: model/seed diversity adds marginal signal. Reused three existing neural seeds and tree/ALS; additionally three new shallow LightGBM stacker seeds on early day5. Full V53 replicas not refitted.
- Runtime: 4.344 seconds locally on CPU.
- Best shared configuration: new_three_stacker_seeds_w0.05. day5: 0.812626070 -> 0.812640324 (+0.000014254); day6: 0.825858710 -> 0.825843902 (-0.000014808).
- Decision: reject; fails +0.0005 on both days. Full cohort/holdout not evaluated.
- Details: artifacts/v54_abcd_D.json. Final audit/report: results/v54_abcd_final_report.md.

A–D retained cumulative gain = zero (< +0.001): stop per user instruction. Practical plateau for tested feature/model family; no mathematical ceiling established. Baseline unchanged; no submission generated.

## 2026-10-29 — Test-like history weighting baseline

- Target history mix: 1.26%/16.37%/82.37%; exact target/source weights. Full selection reweighted V53 AUC 0.813119135/0.820037962. No model fitted. Step1 runtime not separately captured.

## 2026-10-29 — Distribution batch step2

- Hypothesis/change: Exact original 190-feature V53 logistic final head: StandardScaler, C=.01, same .75 parent/.25 head rank blend; parent experts frozen. Weight0 unweighted matched control; weight1 matches target; 2/4 additionally multiply <=5 histories. No new stacker architecture.
- Runtime: 21.516s local CPU.
- Best shared configuration: original_V53_final_head_weight1. day5 weighted 0.810540397->0.810161487 (-0.000378910); unweighted 0.812626070->0.812243119 (-0.000382951); day6 weighted 0.821453597->0.821472739 (+0.000019142); unweighted 0.825858710->0.825784315 (-0.000074395).
- Decision: reject, fails +0.0005 weighted gain on both days. No holdout or submission.

## 2026-10-29 — Distribution batch step3

- Hypothesis/change: Feature counts frozen strictly before each training/validation day. Whole earlier training-window click matrix (not truncated history), Bayesian shrunk full-click item-factor profile. Dedicated short model fits only <=5 histories; context model includes smoothed joint item CTR and user bucket propensity. Monotone probability mapping fits earlyday5 only; output blended only inside named gate.
- Runtime: 7.906s local CPU.
- Best shared configuration: short_only_w0.05. day5 weighted 0.810540397->0.810523989 (-0.000016408); unweighted 0.812626070->0.812613217 (-0.000012853); day6 weighted 0.821453597->0.821349328 (-0.000104270); unweighted 0.825858710->0.825796768 (-0.000061942).
- Decision: reject, fails +0.0005 weighted gain on both days. No holdout or submission.

## 2026-10-29 — Distribution batch step4

- Hypothesis/change: Feature counts frozen strictly before each training/validation day. Whole earlier training-window click matrix (not truncated history), Bayesian shrunk full-click item-factor profile. Dedicated short model fits only <=5 histories; context model includes smoothed joint item CTR and user bucket propensity. Monotone probability mapping fits earlyday5 only; output blended only inside named gate.
- Runtime: 3.453s local CPU.
- Best shared configuration: bucket3_only_w0.05. day5 weighted 0.810540397->0.810411031 (-0.000129366); unweighted 0.812626070->0.812485878 (-0.000140192); day6 weighted 0.821453597->0.821375399 (-0.000078198); unweighted 0.825858710->0.825767494 (-0.000091216).
- Decision: reject, fails +0.0005 weighted gain on both days. No holdout or submission.

Retained cumulative weighted gain = 0 (<+0.001). Stop per user. Detailed features, zero-history probes, limitations and reproduction: results/v54_distribution_final_report.md.

## 2026-10-29 — Phase1 diagnostics only (previous plan stopped)

- No training/model changes/submissions. Full saved selection cohorts; holdout not opened/scored.
- Runtime: 31.781999999890104s diagnostic computation plus report generation.
- Main results: all-time exposure-popularity AUC0.523103/0.521470; last3 AUC0.522827/0.520886; older-age AUC0.505028/0.505347; ALS AUC0.757775/0.760686. No cold/recent items in validation/evaluation. Negatives match popularity-skewed exposure, not uniform catalog sampling. Evaluation has longer prediction horizon and less recently active users.
- Same-frozen-V53 multi-day drift is unavailable; no fabricated drift curve. Rank residual correlation is scale-confounded.
- Decision: diagnostics complete, pause; all closed modeling directions remain closed. Detailed distributions and limitations: results/v54_phase1_diagnostics_report.md.

## 2026-10-29 — Horizon baseline cohort frozen

- Cutoff 1737590400; 49420 first-user events, 9878 sampled; 735 reserved-first events excluded without substitution. Joint raking margins verified. Global click-containing share 62.934035%.
- Baseline models not yet refitted; no horizon AUC or Idea1 gain claimed. Sources/protected hashes and dependency inventory isolated under artifacts/v54_horizon. No holdout opened; no submission.

### Horizon rebuild implementation checkpoint

- Prefix1 original V6 initialization statistics completed in29.375s. Original warm16 ten-epoch seed151 prefix1 fit completed if its checkpoint exists (see isolated timing JSON). No post-cutoff fit labels used. Early initialization clamped to available cutoff to remove original max(cut-2,2) lookahead at cut1.
- Warm refresh kernel implemented and six independent checks pass; no score/gain reported. Full causal V53 ensemble rebuild remains pending; do not substitute ALS-only or original day5/day6 predictions as this baseline. Protected29 files passed checksum comparison.

## 2026-10-29 — Matched warm-user ALS refresh

- Hypothesis: new event-time history changes stale warm ALS vectors. Frozen anchor16/half-life2days; shared item factors and biases unchanged. Standalone ALS, then fixed95% existing warm16 expert/5% ALS blend.
- 49,420 first-event users;44,627 warm;604,857 warm history records all already represented in fitting ledger. Zero post-cutoff history clicks, zero refreshed vectors. Candidate/control scores exactly identical.
- Full weighted overall ALS0.724784897273; simplified blend0.755280328306; both gains0.000000. All arrival/history subgroup deltas0; paired95% intervals[0,0].400 actual user bootstraps of screen overall/late verify degeneracy.
- Scoring8.734s. First metric report pass hit2e-12 tolerance mismatch; reporting-only recovery reused saved scores and relaxed replay tolerance to1e-10.
- Decision: flat by exact identity, stop Idea1; do not schedule full V53 reconstruction from this result. Full V53 remains unbuilt; no holdout/submission. Pause before Idea2/3. Details: results/v54_horizon_matched_refresh_report.md.

## 2026-10-29 — Idea2 event-click measurement only

- Fixed shallow LightGBM fit on43,999 pre-cutoff historical first-user events;10 allowed prediction-time inputs. Full frozen cohort49,420,20% sample9,878.
- Weighted event AUC0.627487 (95% user-bootstrap0.622343–0.632396); arrival1/2/3/4=0.631227/0.622627/0.619314/0.596920. Sample0.622135. Runtime29.141s including causal prefix1 ALS checkpoint; no hyperparameter search.
- Existing simplified-blend event maximum AUC0.584023. Saved V53 selectionday5/day6 event AUC0.621727/0.623142 on different cohorts; propensity partly captured already. No item-AUC gain tested.
- Measurement gate passes; PAUSE before combined event/item probability model. Protected29 hashes unchanged, label invariance check passed, no holdout/submission. Report:results/v54_horizon_event_measurement_report.md.
- User requested same small-model rerun on Kaggle; private minimal frozen-feature bundle prepared, cloud status pending.

### Idea2 measurement rerun on Kaggle, user requested

- Private minimal dataset:riazahmadbutt/techarena-frozen-event-measurement. Frozen features reused; identical shallow event model actually retrained on Kaggle CPU. No new factor fits on Kaggle.
- Notebook:https://www.kaggle.com/code/riazahmadbutt/techarena-idea2-event-click-measurement; saved version1 with outputs. Model/measurement6.138s, total cell13.509s. Python3.13.15,LightGBM4.6.0.
- Weighted event AUC0.627487218457; all subgroup AUCs reproduce local results. Maximum prediction difference1.11e-16. SHA256 bundle verified on remote runtime.29 protected artifacts rechecked unchanged.
- CPU session stopped. No holdout, no submission, no product candidate. PAUSE for review after step1. Raw cloud output:artifacts/v54_horizon/event_measurement_001/kaggle_report.json.

## 2026-10-29 — Idea2 Step2 pre-registered matched probability test

- User explicitly approved one exception to closed calibration/stacker line. Fixed logistic then boosted models; no search. Runtime estimated2–5compute minutes plus Kaggle startup.
- Historical CDFs and positive softmax temperature for conditional item choice; chronological pre-validation Platt event calibration. Same conditional-choice scores in every arm; control constant event factor versus learned event factor. Stronger control includes max/mean score,user click rate,history length.
- Frozen horizon cohort and IPF weights preserved. Separate actual V53 diagnostic: fit day5 user-OOF calibration inputs, test allowed day6 only. Legacy score ranks inherited; no full V53 horizon claim.
- Frozen plan and minimal datasets:artifacts/v54_horizon/event_product_001. Kaggle private input dataset created; CPU execution queued. No holdout/submission/protected writes.

### Idea2 Step2 completed on Kaggle CPU

- Fixed logistic then boosted; historical CDF/temperature plus earlier-block Platt calibration; same conditional choice in all matched arms. Main run157.020s.
- Horizon choice-only0.752211990 -> logistic0.755326063 (+0.003114073,95%0.002626556–0.003533568); boosted0.755036395 (+0.002824405,95%0.002348530–0.003304004).
- Beyond strengthened controls: logistic+0.000909074 (95%0.000676759–0.001152789); boosted+0.000711270 (95%0.000510908–0.000892471). Sample gains+0.000659820/+0.000144461; boosted fails screening.
- Saved V53 day6 strengthened gains+0.000457294/+0.000323320, but all replacements lose to untouched V53: best0.818817715 vs0.820037962 (-0.001220247). No promotion. Horizon logistic-augmented is only+0.000274129 versus original simplified blend and below it on20% sample.
- Global and arrival/history AUCs and paired-user intervals in results/v54_horizon_event_product_report.md; raw cloud report and original-reference bootstraps under artifacts/v54_horizon/event_product_001. Probability calibration transport imperfect and legacy saved-V53 rank caveat documented.29 protected checks passed. No holdout/submission/evaluation scoring. PAUSE.

## 2026-10-29 — Lever 1 diagnostic: is there a weekday/weekend gap? (no model change)

- **Hypothesis:** evaluation rows are 97% weekdays while all selection/final days are weekends (Jan 24–26), so a weekday-trained ensemble might transfer better.
- **Change:** none. Two read-only checks, 2 minutes of CPU. (a) Behaviour by calendar day in `artifacts/train.npy`: event count, CTR, click rate when the candidate is in the history, share of candidates in history, share of clicks/candidates on the 100 most-clicked items, mean history length, evening share. (b) The saved V60 holdout (`artifacts/v60_weekday/predictions.npz`, one model, one cutoff at Jan 24 00:00) split by arrival day.
- **Results:** (a) No weekday/weekend difference at all: CTR 0.0626–0.0632 every day, click-given-in-history 0.287–0.299, candidate-in-history 0.0256–0.0262, top-100 click share 0.020–0.021, evening share 0.398–0.402, events 97.6k–98.5k per day. Only history length grows monotonically with the calendar (16.3 → 17.6 clicks), which is history accumulation, not day type. (b) V60 holdout ordinary/adjusted AUC: Fri (arrival day 1) 0.8035/0.7986, Sat (day 2) 0.7794/0.7499, Sun (day 3) 0.7721/0.7454 — the same decay by arrival day seen in every earlier horizon study; it reflects staleness/activity of late-arriving users, which the adjusted metric's arrival weighting already covers.
- **Decision:** the weekday/weekend gap is not real; it is an arrival-day gap. No weekday refit (Lever 1 step 2 skipped per the user's rule "only if the gap is real"). V53 itself cannot be scored on a weekday without refitting its cutoffs, so the one-model V60 holdout was used as the matched evidence. Cumulative kept gain this round: 0.

## 2026-10-29 — Lever 2 screen: session-structure timing features on the V53 head (v70_session_timing.py)

- **Hypothesis:** timing features had been judged on the old single-day metric; session structure (gaps > 4 h, session count, last-session size/share, click rate per day, candidate clicked in last session, one-day-decayed candidate repeat score) had never been tried at all.
- **Change:** 12 strictly-past timing columns appended to V53's 190 head features; same head (C=0.01, seed 4801), same 0.75/0.25 rank blend, same day-5 user cross-fit then day-6 future fit. Baseline = V53. Runtime 75 s local CPU.
- **Results (adjusted AUC):** day 5 0.810911 → 0.810925 (+0.000014, interval [+0.000001, +0.000026]); day 6 0.817169 → 0.817191 (+0.000022, interval [+0.000012, +0.000033]). Ordinary AUC +0.000016 / +0.000024. Files: `artifacts/v70_session_timing_{frozen,report}.json`, `_day{5,6}.npz`, `_models.pkl`.
- **Decision:** reject — ten times below the +0.0002 screening rule and twenty times below the +0.0005 confirmation rule. Levers 1 + 2 combined gain ≈ 0 (< +0.003), so the round's stop rule triggers; Lever 3 not started. Not promoted; V53 unchanged.

## 2026-10-29 — nx round, idea E: short-history re-score of the saved slate-composition residual (nx_e_slate_short.py)

- **Hypothesis:** the V58 slate residual (other nine candidates as context) was flat overall (+0.00003) but might carry taste for
  cold users (0–4 history clicks), which the overall number would hide.
- **Change:** none fitted. `artifacts/v58_slate_residual_day6.npz` (slate arm, matched zero-slate control, both as raw residual and
  as the frozen 10% rank blend with V53) re-scored per history bin on full day 6; additionally the slate/control blend substituted
  for V53 only on the 0–4 bins at weights 0.1/0.25/0.5 and scored with the project adjusted AUC. Runtime 20 s CPU.
- **Results (ordinary AUC inside the bin):** bin 0 (198 events): V53 0.5221, slate blend 0.5204, control blend 0.5177, raw arms
  0.513/0.501. Bin 1–4 (2,357 events): V53 0.747242, slate blend 0.747240, control blend 0.747274, raw arms 0.696/0.695.
  Bin 0–4: slate blend − control blend = 0.000000. Short-only substitution, full adjusted AUC change: −0.000016 (w 0.1),
  −0.000115 (0.25), −0.000908 (0.5); the control arm gives the same numbers within 1e-5.
- **Decision:** reject; the candidate-set information adds nothing for short histories either. Output `results/nx_e_slate_short.json`.

## 2026-10-29 — nx round, idea F: item trend + transductive cohort-exposure features (nx_f6_trend_screen.py)

- **Hypothesis:** research family F6 (item lifecycle/trend, label-free exposure share inside the scored cohort's own slates, which is computable over evaluation.csv).
- **Change:** 12 item features (cohort exposure share overall and within ±2 h, pre-cutoff exposure share and smoothed CTR over all/1 d/6 h, three trend log-ratios, slate rank of cohort share); small LightGBM on [logit V53, features] vs matched control [logit V53], 2-fold user cross-fit per day. Runtime ~3 min CPU.
- **Results:** every raw feature has adjusted AUC 0.497–0.530. Matched gain (f6 arm − control arm): day 6 +0.000235, day 5 −0.000319.
- **Decision:** reject. Item-level popularity/trend dynamics carry no usable signal in this simulator.

## 2026-10-29 — nx round, diagnostic: are slates personalized? (no model change)

- 60k random events: candidates found in the user's own 20-click history 0.258 per event vs 0.0050 expected under popularity sampling (52×). Consecutive events of the same user share 0.0335 candidates vs 0.0031 for random other-user pairs (11×).
- Interpretation: the logging policy is strongly user-dependent, but the exposure-factor screen below shows the extra information is mostly history-driven (history items injected into slates), not latent taste.

## 2026-10-29 — nx round, idea C: exposure-policy factorization screen (nx_exposure.py --day 6, first run)

- **Change:** TruncatedSVD (k=32) of the pre-cutoff user×impression matrix (exposures as POSITIVE taste evidence, unlike imp15 which treats them as click-model negatives). Runtime ~1 min.
- **Results (day 6 adjusted):** best `expo_typ` raw 0.5635 (`expo_cos` 0.566; click-SVD reference 0.639); rank blends +0.000303 (2%), +0.000332 (5%), −0.00066 (10%). 0–4-history AUC 0.7384 → 0.7392.
- **Decision:** screen pass on day 6 only (+0.0002 < gain < +0.0005); not run on day 5. Candidate for use as head features rather than a rank blend. Report: `artifacts/nx_exposure_screen_day6_report.json`.

## 2026-10-29 — nx round, idea A: SASRec screen result (read from artifacts/nx_seq_screen_day6_log.json)

- d=32, L=30, 1 pretrain + 1 joint epoch, 457 s. Standalone raw adjusted 0.5833; every V53 blend loses (5%: −0.00071).
- **Decision:** kill criterion (< 0.76) met. Caveat: the "candidate in history" flag alone scores ≈0.60 and click-SVD 0.64, so 0.58 indicates an underfit or defective model, not evidence against sequence modelling.

## 2026-10-29 — nx round, idea G: event-mass correction (nx_event_mass.py, nx_event_mass_combine.py)

- **Hypothesis:** global row AUC depends on which events contain a click; V53's per-event mass looked weak (event AUC 0.563).
- **Correction of the premise:** 0.563 was measured on V53 `pred`, which is a uniform rank score (event sums average 5.0); per-event sums of ranks are not meaningful. On V53 `calibrated` probabilities the event AUC is 0.681 (day 6) / 0.675 (day 5); the rank transform is monotone over all rows, so global AUC is unaffected. Do not reuse the 0.563 figure.
- **Change:** LightGBM event-click model on 19 causal user-behaviour/history/slate features (prior events and their labels < cutoff, gaps, history recency, candidates-in-history), trained on the 4 days before cutoff (~392k events), 34–66 s. Standalone event AUC 0.6328 / 0.6323. Combination p = V53 calibrated within-event share × logistic(logit V53 mass, logit q); day 6 fitted on day 5, day 5 by 2-fold user cross-fit; share temperature 0.8–1.5.
- **Results (adjusted, t=1.0):** day 6 0.817169 → 0.817029 (−0.00014; −0.00014 vs matched V53-mass-only control); day 5 −0.00068 vs V53, +0.00009 vs control. Combined event AUC does not exceed V53's own (0.679 vs 0.681).
- **Decision:** reject. V53 already captures event-level click propensity; the z-scored first attempt (large losses) was a scale artefact of combining with rank scores.

## 2026-10-29 — nx round, idea B: user-based neighbourhood CF (nx_userknn.py, nx_headscreen.py)

- **Hypothesis:** a user-user kernel over click sets (high-rank, local) adds taste information beyond the rank-16/32 factor
  experts; never tested before (only item-KNN).
- **Change:** causal click ledger (supplied histories + labels < cutoff, unique user/item/time), IDF-cosine user similarity, the
  scored event's own history merged into its query vector, self excluded. Top-100 neighbour scores (click, last-2-days click,
  2-day-decayed click, neighbour count, exposure, neighbour CTR) and all-user scores with similarity powers 1/2/4 (click,
  popularity lift, exposure, CTR). Day 6 only (screen). Runtime ≈ 2.5 min per feature build, 23 s head screen.
- **Results (full day 6):** top-K scores non-zero on 7.2% of rows, raw adjusted AUC ≤ 0.568; all-user scores non-zero on 33.7%,
  best raw adjusted 0.6063 (ucf1_lift). Rank blends with V53: all negative (best −0.000065 at 2%). Cross-fitted logistic head
  (logit V53 + 10 features + interactions, 2-fold user split) vs matched V53-only control: +0.000042 adjusted.
  Short histories (0–4): raw 0.578 vs V53 0.738.
- **Decision:** reject (far below the +0.0002 screen). V53 unchanged.

## 2026-10-29 — Review of user latest local changes (no new experiment)

- Read existing nx results and source. Sequence screen2 standalone adjusted AUC0.651924 versus0.583275 before, but best V53 blend0.816741 versus0.817169 baseline (-0.000427673).
- Latest user-neighbour screen2 best blend0.817115 (-0.000054112); event mass t1 loses0.000679625/0.000140364 on days5/6. Exposure best existing day6 gain+0.000332051 only; no two-day confirmation and below+0.0005.
- Metadata-only cohort check: latest nx scorers use full day6,35,138 events, including10,716 outside allowed24,422 selection events. Not the frozen multi-day test-like validation. Do not call these clean confirmation under the agreed protocol. No reserved labels opened/scored in this review.
- Open Kaggle measurement is still saved version1, AUC0.627487.29 protected checks unchanged. No models fitted, predictions generated, or submissions created. Details:results/latest_changes_review_20261029.json.

## 2026-10-29 — nx round (Idea-A improver session, nxa_ files), idea C: exposure-policy expert queued (nxa_exposure2.py, nxa_expo_final.py)

- **Hypothesis:** the simulator's recommender chose every slate from its belief about the user, so the items a user was SHOWN
  before the cutoff (no labels) estimate taste, including for users with few or no clicks.
- **Change:** TruncatedSVD (k=64, seed 4403) of the IDF-weighted log1p user×item exposure matrix from impressions with time <
  cutoff; per candidate the cosine with the user's uniform mean exposure profile (k=16 and 32) and with the SVD user factor (k=32).
  Expert = mean rank of the three. Recipe chosen on day 6 only (5 experts × global rank blend / within-event centred rank /
  within-event centred z): pred = rank(parent) + 0.015 × (expert − event mean) / std. Same script for day 5, day 6 and evaluation
  (cutoffs 1737763200 / 1737849600 / 1737936000). My re-run of nx_exposure.py reproduced the other session's 19:26 numbers exactly.
  Variants rejected on day 6: recency-weighted or last-2-day profiles (worse), k=64 (worse), binary/log weighting (worse),
  co-exposure item space applied to click histories (raw 0.59 but weaker in blend), user-kNN-style combinations.
- **Runtime:** features 71 s (day 6), 60 s (day 5), 42 s (evaluation), 2 BLAS threads; recipe/eval script 40 s.
- **Results (adjusted AUC, project weighted_auc):** day 6 0.817169 → 0.817810 (+0.000641; on V39 0.816661 → 0.817283, +0.000622);
  day 5 0.810911 → 0.811398 (+0.000486; on V39 +0.000474). Paired event bootstrap (400, seed 3): day 6 [+0.00045, +0.00083],
  day 5 [+0.00029, +0.00068]. Every history bin improves on both days (day 6: bin 0 0.522 → 0.528, 1–4 0.7472 → 0.7489,
  20 0.8424 → 0.8427). Raw expert alone 0.569 / 0.564. Ordinary AUC +0.00057 / +0.00045.
- **Decision:** queued for the tester (`tst4_queue.md`, nxa_expo). Day 5 is 0.000014 below the +0.0005 gate; expected verdict
  "screen pass" unless the tester's interval/holdout says otherwise.

## 2026-10-29 — nx round (Idea-A improver session), F1: time-decayed multi-anchor covisitation + exposure-conditioned pair CTR (nxa_covis.py)

- **Hypothesis:** the closed item-item screens missed time locality, multi-anchor rank weighting and exposure normalisation;
  a pair CTR conditioned on the candidate having been SHOWN removes the popularity skew of exposures.
- **Change:** ledger co-clicks of the same user within 24 h (weight exp(−gap/6 h)), degree-normalised, scored over the 20 newest
  anchors with weights 1/(rank+1); pair CTR over other users' training events (< cutoff) for (history item, shown item), smoothed
  (α=20) lift over item CTR, the scored user's own events subtracted. Day 6, 181 s, < 1 GB.
- **Results (day 6 adjusted):** R_time non-zero 9.4%, raw 0.556; pair-CTR raw 0.534–0.560; rank blends 2/5/10% all negative
  (best −0.000067); within-event z blends all negative; cross-fitted head (8 features) −0.000005 vs control.
- **Decision:** reject.

## 2026-10-29 — nx round (Idea-A improver session), idea A: SASRec-style sequence model — closed (nx_seq_core.py, nx_seq_warm.py)

- **Hypothesis:** a causal transformer over each user's full click ledger (histories + pre-cutoff clicks), trained on all
  sequences with a next-click sampled-softmax objective plus the 10-candidate/no-click event loss, adds sequence information
  that the bag-of-history experts (choice model, ALS, SVD profiles) miss.
- **Change / runs (day 6, cutoff 1737849600, fixed seeds):** (1) from scratch, d=32, L=30, 1+1 epochs, 457 s: raw adjusted 0.583
  (optimiser too slow on sparse tables); (2) table lr 1e-2 + user vector, 2+2 epochs, 48 min wall (30-min machine stall inside):
  raw 0.573 → 0.652, best V53 blend −0.00043; (3) matched warm pair from the choice model's initialisation (Adagrad tables):
  bag control raw 0.7512/0.7455, blend −0.000055/−0.000092; transformer raw 0.7389/0.7327, blend −0.000213/−0.000245 (epochs 1/2).
- **Ledger facts:** 1.01 M unique clicks, 18.5 per user (median 14, p90 38): the 20-click supplied window already covers most
  of each user's sequence, so "pretraining on longer sequences" has little extra data.
- **Decision:** close Idea A (transformer worse than its own bag control by 0.00015; hard stop was +0.0002). No GPU/Kaggle job
  requested.

## 2026-10-29 — nx round (Idea-A improver session), C step 2: exposure expert through a small exportable logistic head (nxa_h1.py, nxa_h1_final.py) — queued

- **Hypothesis:** the tester's verdict on nxa_expo (day 5 +0.000486) suggested the fixed rank-space z blend is the bottleneck,
  not the signal; a head that learns how to add the exposure term (in logit-rank space, weights fitted on the PREVIOUS day's labels)
  should add it more efficiently, and stays exportable because its only parent input is V39 on evaluation rows.
- **Change:** LogisticRegression(C=0.01) on [logit(rank(parent)), E3z] (E3z = within-event centred nxa_expo expert / std), 5-bin
  evaluation-mix weights; day 6 fit on full day 5, day 5 by 2-fold user cross-fit (seed 9901), evaluation fit on full day 6 with V39.
  Selection on the day-6 SELECTION subset only (grid saved: artifacts/nxa_h1_selection.json): F0 control 0, F1 +0.000803,
  segment-recalibration control S0 (no exposure) +0.000178, F2b (segment-specific exposure strength) +0.000811, F2 +0.000999,
  F3 +0.000976. F2/F3 rejected: their extra over F1 equals S0, i.e. recalibration (closed line), not the new signal.
  Frozen in artifacts/nxa_h1_frozen.json before any day-5 evaluation. Other day-6-selection checks the same hour (all saved,
  artifacts/nxa_c2_feature_screen_sel6.json): one-step policy MLE, exposure-space neighbour CTR and choice-model catalogue rank
  (research F2) added ≤ +0.00006 to E3 or only with counter-intuitive signs — not used.
- **Runtime:** head fits 23 s; features already built.
- **Results (adjusted AUC, project weighted_auc):** V53 parent: day 6 0.817169 → 0.818023 (+0.000854; selection subset +0.000803),
  day 5 0.810911 → 0.811431 (+0.000520). V39 parent: day 6 +0.000819, day 5 +0.000491. Paired bootstrap (400): V53 day 6
  [+0.00053, +0.00116], day 5 [+0.00023, +0.00084]. Evaluation head coefficients [1.277, 0.105] (day-6 fit) vs [1.236, 0.092]
  (day-5 fit): stable. Evaluation output rank-transformed, Spearman 0.9957 with V39.
- **Decision:** queued as nxa_h1. Passes +0.0005 on both days against V53; against V39 day 5 is +0.000491.

## 2026-10-29 — Codex independent latent taste-group screen

- New hypothesis: discrete latent preference groups share item-response residuals beyond ALS.16 fixed groups, Bayesian strength20, global user/item propensities removed, all scored-user counts excluded. No search; historical-only feature CDF.
- Frozen test-like20%user cohort9878, IPF history+arrival weights. Runtime13.328s.0.759031195->0.759227170 (+0.000195975;paired95%[+0.000017548,+0.000336645]). Arrival1/2/3/4 gains+0.000107791/+0.000578893/+0.000295171/-0.000031029.
- Reject fixed screen; fails+0.0002 overall/every-arrival rule. No rounding up, full confirmation, tuning, holdout, evaluation or submission.29 protected files unchanged. Details:artifacts/v54_horizon/cdx_taste_groups_001/report.json. Own journal:results/cdx_research_journal.md.


### Codex independent graph preference screen — 2026-10-29
Corrected matched run 692.875s, weighted AUC0.759031195->0.759027097 (-0.000004099), paired95%[-0.000010819,+0.000003880]. Arrival gains1–4:-0.000007692/+0.000007754/+0.000012331/-0.000002581. Standalone bag0.711418694/graph0.711085878. No survivor, no full confirmation or sweep. Protected29 unchanged; saved predictions reproduced; label-mutation check passes in both arms. Complete report results/cdx_independent_preference_report.md.

### Codex probabilistic exposure screen — 2026-10-29

- New structural hypothesis: hierarchical Gamma-Poisson model of actual shown-item counts captures targeting beyond IDF-log SVD.32 factors,80 fixed iterations,seed8221,source Gopalan/Hofman/Blei2015. Own variational implementation; monotonic ELBO and exact responsibility mass checks. No click labels in fitting, no artificial warm-up exposures or new head.
- Frozen9878-user horizon cohort, evaluation-mix history/arrival IPF weights. Prefix2 historical CDFs; prefix3 frozen models across all4arrival days. Matched fixed0.015 exposure bridge to simplified horizon blend, HPF expert replaces SVD expert. This is not fullV53/nxa_h1 or official scoring.
- Runtime154.704s (pilot upper estimate454.7s). Original baseline0.759031195; SVD control0.759536768; HPF0.759292110. Difference vs SVD-0.000244658, paired user-bootstrap95%[-0.000677232,+0.000146089]. Arrival gains versus SVD1–4:-0.000286509/-0.000006632/+0.000021294/-0.000599927.
- REJECT. HPF gains+.000260915 vs original overall but fails against SVD and loses on arrival4; no sweep or full confirmation. Protected29 unchanged; no holdout/evaluation/submission. Artifacts:artifacts/v54_horizon/cdx_policy_hpf_001/report.json. Explicit learned logging-density step now testing separately; see own journal.

## 2026-10-29 — Round result (lead): nxa_h1 ACCEPTED by tester; submission_v41_candidate.csv

- **Accepted candidate:** nxa_h1 = small logistic head over [logit(rank(parent)), within-event-centred exposure-policy expert E3z] (scripts `nxa_exposure2.py --day 5/6/7`, `nxa_h1.py`, `nxa_h1_final.py`; frozen recipe `artifacts/nxa_h1_frozen.json`, selection grid `artifacts/nxa_h1_selection.json`).
- **Tester recomputation (adjusted AUC gains, paired 95% intervals):** day 6 full +0.000854 vs V53 [+0.00053, +0.00116] (+0.000819 vs V39); day-6 selection +0.000803; day 5 +0.000520 vs V53 [+0.00022, +0.00084] (+0.000491 vs V39); fresh holdout +0.000974 vs V53 [+0.00042, +0.00157], +0.000928 vs V39. Leakage audit clean; protected files unchanged.
- **Submission:** `submission_v41_candidate.csv` = recipe applied to V39's evaluation predictions; 529,390 rows in evaluation.csv order, values in [0,1]; SHA-256 71ada3f0e16240526fb61ce4846d29dd93bd4a9369001d0795e21902de618756; Spearman 0.9957 vs submission_v39.csv.
- **Rejected this round:** v70 session timing, sequence transformer (Idea A), user-user CF (B), slate for cold users (E), event-mass (G), co-visitation (F1), catalogue-rank/one-step policy extras. `submission_v40_candidate.csv` (nxa_expo, user-override exploratory file) is superseded by v41.
- **Caveats:** day 5 uses user cross-fit (as V53 did); on the V39 parent day 5 is +0.000491; the expert's columns were picked on full day 6 (5 near-identical options). Expected official gain over V39 ≈ +0.0005 to +0.0009; V39 itself not yet scored.

## 2026-10-29 — nx round phase 2 (Idea-A improver session): exposure features inside the tree + richer exposure features — stop rule

- **Hypothesis:** as with warm16 (+0.010 as tree features), the exposure-policy signal would gain more inside the main tree than
  through the nxa_h1 head; richer exposure views (audience, exposure-minus-click contrast, recency, typicality percentile) add more.
- **Change:** (1) 8 label-free exposure columns for every cached block (`artifacts/features/nxa_expo_{day2..day6,val5,val6,eval}.npy`,
  impressions < block cutoff), appended to imp19's 354 columns (layout verified: imp19_tree_d6.txt replays val6 exactly);
  matched LightGBM screen with imp19 SETTINGS/seed, 735 trees, days 3–5, the same 20% of events (seed 7401), control vs treatment.
  (2) six richer cohort features added to the nxa_h1 head (fit on day 5). Both scored on the day-6 SELECTION subset only; grids
  saved (`artifacts/nxa_tree_screen_eval.json`, `artifacts/nxa_rich_screen.json`). Runtime: features 2 min, trees 4.6 min each.
- **Results (selection subset, adjusted):** raw tree 0.810576 → 0.811066 (+0.00049), but vs nxa_h1 the treatment tree adds
  +0.00001 (10% blend) and −0.00004 (cross-fitted head) relative to the control tree; richer features −0.00005 … +0.00002 each,
  −0.000017 together.
- **Decision:** both fail the +0.0002 screen; together ≈ 0 vs the +0.0008 stop threshold → stop rule triggered. nxa_h1 remains
  the best (full day 6 0.818023). No candidate queued; no full tree fit run.


### Codex learned logging-policy screen — 2026-10-29

Learned logging-policy factors, genuine shown triples vs independent uniform noise; no click-label fitting. 32dims,3fixed epochs,seed8231,Adagrad0.03/acc0.1,anchor0.01; same historical-CDF0.015 bridge/SVD control as HPF. Runtime38.672s. SVD0.759536768->0.759759776 (+0.000223008;paired95%[+0.000025950,+0.000410513]), original baseline0.759031195. Arrival gains1–4:+0.000317588/+0.000002686/+0.000079195/-0.000071857; 3+only+.000004984; history1–5-.000211430. REJECT under all-arrival gate; no full confirmation, tuning or promotion. Protected29 unchanged; manifests/replay/label and order invariance pass. No holdout/evaluation/submission. Not a fullV53/nxa_h1 or official-score gain. Report:results/cdx_policy_followup_report.md.


### Codex 30 October — collective and multi-interest screens

Matched collective exposure/click item factors: original warm16 parameters reproduced bit-exact in control. Same16dims/10epochs/seed151/warmaux1/lr.045/reg.002; exposure target actual impressions vs uniform noise, weight.1/seed8531, only treatment shares V data gradients. Runtime311.719s. Frozen20%horizon weighted0.759031201->0.758751566 (-0.000279634;paired95%[-0.000410215,-0.000149736]); all4arrivaldays lose. Separate no-training two-interest diagnostic using frozen H-centroids, conditional-choice mixture with unchanged event mass:0.757435768 (-0.001595433;paired95%[-0.002749275,-0.000406804]),44.594s. Both REJECT; no full confirmation or tuning. Same frozen95%warm-rank+5%ALS bridge, not fullV53/nxa_h1. Sources Singh/Gordon2008 and Li etal2019MIND motivation; own adaptations, no external code copied. Replay/label invariance/timestamps/order/mass/weight audits pass;29protected unchanged. No holdout/evaluation/submission.0.82 target unmet; reported best local0.818023 unchanged. Report:results/cdx_oct30_preference_report.md. Actual competition URL requested for further research.

### Codex source lookup — description and screenshots, 30 October 2026

User supplied the CTR description and screenshots; visible leaderboard top is free lunch 0.832869 (screenshot, not live scoring). Instructions and Starter Solution links are shown but their destinations/content and page URL are absent. Inspected public UK2026 and linked UK2025 sites; they concern Career Survival Agent/Cangjie systems programming, not this CTR challenge. No unrelated contest rules applied. Requested actual CTR address-bar URL or Instructions text. No training, model changes, holdout/evaluation scoring, submission or official-score update. Static V53 source check confirms full horizon reconstruction is still absent and depends on six explicit expert sources plus a three-seed score mean and V39 parent. No runtime pilot begun. Details: results/cdx_competition_source_status_20261030.json.

### Codex metric reconciliation and user clarification — 30 October 2026

User clarified that the task is already supplied; stop repeatedly requesting the competition URL and proceed with supplied data under existing restrictions. Read-only verification of saved nxa_h1 predictions on ALLOWED selection events only (day5 35,230, day6 24,422); never access full saved outcome keys. No holdout scoring or evaluation-file reads. On day6 selection: V53 parent AUC0.817527526 under existing5bin weights /0.820037962 under supplied3bin mix; current H1 with V53 parent0.818330578 /0.820806499. On day5: current H1 with V53 parent0.811430886 /0.813633495. Exportable V39-parent H1: day6 0.817768856 /0.820240397, day5 0.811017168 /0.813218048. The historical reported0.818023 is full-day5bin, so differs in weighting AND cohort. These are reconciled scores of the SAME saved model, not new gains or proof of official0.82. Last confirmed official0.816623 unchanged. No model fit, submission, promotion or official-ledger change. Report: results/cdx_current_metric_check_20261030.json.


### Codex directional factor uncertainty — 30 October 2026

Frozen warm16 means/item/history factors, prior empirical U covariance plus actual pre-cutoff slate expected Fisher; 32 positive spherical cubature points, fixed original event-click mass. All fits pre1737590400; no validation-outcome fitting. Runtime22.25s. Sample9878 weighted0.759031201 ->0.758659198, delta-0.000372002; paired200-user-bootstrap95%[-0.000914095,+0.000150219]. Arrival1–4 deltas:-0.000342035/-0.000902415/+0.000840688/+0.000170323. History1–5 loses-0.001772388. REJECT, no full confirmation/tuning. Approximation is not full Bayesian MF; omits item/history posterior coupling and warm auxiliary curvature. Numerical Hessian/PSD/quadrature/zero-covariance checks and29protected hashes passed. No holdout/evaluation/submission. Source ideas: Salakhutdinov/Mnih2008 BPMF; Coscrato/Bridge2022 uncertainty; own implementation. Artifact:artifacts/v54_horizon/cdx_uncertainty_001/report.json.


### Codex catalog AE moved to Kaggle —30October2026

User explicitly requested Kaggle instead of local training. Stopped local run at12DAEepochs withCtrlC before any validation scoring. Prepared25.329MB self-contained, manifest-hashed bundle of pre-cutoff deduplicated click matrix, fixed initial factors, label-free frozen horizon inputs and separately stored permitted validation labels; no holdout/evaluation/submissions. Initial automatic upload review rejected insufficiently specific export consent; described exact payload/account/private destination, user delegated best choice, selected private GPU option; subsequent same upload flow accepted. Created PRIVATE TechArena Codex Catalog AE Inputs dataset and isolated TechArena Codex Catalog AE Matched Test notebook, original measurement notebook untouched. Saved Version1 Fixed DAE VAE GPU screen; UI confirms Running withGPU T4x2,17hours quota previously visible. Own CUDA adaptation restarts both fixed arms from scratch, unchanged epochs/dims/seed/objective/bridge/gates; GPU pilot must stayunder30min. GPU run pending; no AUC/result claim. Status:artifacts/v54_horizon/cdx_catalog_ae_001/kaggle_run_status.json.


KaggleVersion1(id363747554) GPU TeslaT4 confirmed. Pilot112.197s conservative estimate,20DAEepochs about1.07s/epoch. Inference failed before any validation outcome access: own AST export omitted @torch.no_grad onscore, causing numpy-requires-grad error. V1 retained; restore original inference wrapper in notebook only, identical models/data/settings and restartbotharms. This is a porting repair before metrics, not model selection. GPU repair manifest:cdx_catalog_ae_001/gpu_inference_repair.json.


### Codex catalog autoencoder survives on Kaggle —30October2026

Private KaggleVersion2(id363749125),TeslaT4. Fixed matchedDAE/VAE20epochs16dims, no validation tuning. Actual model+scoring+bootstrap86.417s, totalnotebook118.3s. DAE screen9878users:0.759536768->0.761009521(+0.001472753;95%[+0.001275695,+0.001652786]); all4arrivaldays+.0002pass. Full49420:0.755988900->0.757449150(+0.001460249;95%[+0.001367337,+0.001552716]); arrival1–4gains+.001316801/+.001674831/+.002159761/+.002088841; all+.0005pass. VAE full+.001444402; VAE-vs-DAEscreen-.000023493: no extra benefit, retain simplerDAE survivor. Fullhist0/1–5/6–20gains-.000096049/+.003205789/+.001261584; report zero-history loss. Fullcohort includes screen, no independentholdout. Survivor ONLY against simplified exposure-enhanced causal horizon reference; NOT measured fullV53/nxa_h1 gain and MUST NOT add delta to0.818023. Official0.82 unproven; strongest protectedmodel notpromoted. Independent local read-only audit: AUC reproducedwithin2.7e-12, labels/order invariance, sameeventmass, cutoff/history/weights/cohortchecks,29protectedunchanged. No holdout/evaluation/submission. Report:results/cdx_catalog_ae_report.md. Models/predictions/report downloaded and receipt/manifests saved undercdx_catalog_ae_001/. Compute preference: future heavyfits onKaggleGPU when supported. Pause after confirmedcontrol result.


### Codex stronger-parent DAE package ready; browser stopped —30October2026

User requested faster improvement toward0.82. Preregistered the already-surviving DAE, unchanged16D/20epochs/Adam.001/batch256/dropout.5/seed8711, with original same-day warm16seed7 initialization, against frozen H1/V53 primary, H1/V39 and V53 secondary. Fixed5percentagepoint rank replacement; no blend/epoch/seed search. NewDAEfits strictly before respective day5/day6cutoffs; existingday5H1user-crossfit calendar explicitly documented,day6headtrainsearlierday5. No newhead or evaluationstatistics. Frozenuserhash20%screens7017/4896, fullallowedcohorts35230/24422. Supplied3binmixprimary and historical5binguard; screen+.0002BOTHdays/bothweightings, onlysurvivor full+.0005BOTHdays/bothweightings. Source/data manifests and49.861MB privateKagglebundle atcdx_dae_strong_001/. No localtraining. Estimate2–4min onT4basedpreceding1s/epochpilot.29protectedunchanged. In-appbrowserCDP stalled; reset/fresh-tab/visibility recovery unsuccessful. DocumentednativeChrome fallback opened a clean new tab; tool then explicitly reported userphysicalEscape STOP. All ComputerUse stopped immediately. No payload uploaded, no newKagglejob, no newAUC, no holdout/evaluation/submission. Preparedwork retained; do not repeat or refreeze.

## 2026-10-30 — nxb round (protocol_v4 improver, nxb_ files): Track A clean-room baseline + Track B converged choice model

- **Track A hypothesis:** the 60-version chain misses a simple, well-understood signal that a clean LightGBM on ~60 causal
  features would expose. **Change:** new self-contained code (`nxb_a_features.py`, `nxb_a_fit.py`; raw arrays only): 65 features
  per day block (item/user CTRs overall/1d/3d/time-bucket, activity, prior exposures, candidate-in-history count/position/recency,
  history length/recency, co-click cosine sum/max/recency/last, idf click-SVD-32 history and ledger profiles, exposure-SVD cosine,
  within-event rank/minus-max/minus-mean, event stats), all from rows < block cutoff; LightGBM (63 leaves, lr 0.05, seed 6201);
  day 6 <- blocks 2..5, day 5 <- 1..4; trees chosen on the day-6 SELECTION subset (grid `artifacts/nxb_a_a1_grid_day6.json`).
  **Runtime:** features 50-95 s per block; fit 200 trees ~40 s (1,500-tree grid 346 s).
  **Results (adjusted):** day 6 full 0.730629 / selection 0.730864 / plain 0.736411; day 5 0.721584 — vs nxa_h1 0.818023 /
  0.818331 / day 5 0.811431. Per history bin the chain wins every bin by 0.06-0.09 except 0 clicks (198 events: clean 0.563 vs
  nxa_h1 0.531 on day 6, 0.583 vs 0.521 on day 5). Best blend with nxa_h1 (selection subset): rank 2% = +0.000008; clean model
  only inside 0-history events: +0.000004. **Decision:** no candidate. Diagnostic: nothing simple is missing; the chain's value is
  the trained choice factorisation (warm16 alone 0.8015 plain day 6) — simple statistics stop at ~0.73.
- **Structural diagnostics (no model change):** history-profile cosine in a click-SVD space has a sharp knee at rank 16 (k 12
  0.697, 16 0.724, 24 0.723, 32 0.709, 64 0.695, 256 0.649; plain AUC day 6) -> the simulator very likely uses 16-dim factors;
  oldest-10 history clicks are as informative as the newest 10 (0.713 vs 0.715; no taste drift); same-day leave-one-out item CTR
  is no better than previous-day CTR (0.520 vs 0.519; no item drift); event click rate does not depend on the delay of a user's
  first event after a cutoff (0.63-0.65). Scripts `nxb_diag_svdk.py`, `nxb_diag_recency.py`, `nxb_diag_drift.py`.
- **Track B hypothesis:** a candidate-aware choice model (DIN attention over the 20 timestamped clicks, user/item embeddings,
  11-way softmax) trained to convergence with early stopping on the pre-cutoff day reaches >= 0.80 standalone.
  **Change:** `nxb_b_seq.py` (torch, sparse Adagrad 0.045, d=16, click-SVD warm start, warm-aux loss on first-event histories,
  optional attention / time-gap / repeat / exposure / previous-impression terms); phase 1 trains < day 5 and early-stops on the
  day-5 first-per-user cohort; phase 2 refits < day 6 for the chosen epochs. 12 variants logged in `nxb_journal.md`.
  **Results:** best = plain bag model with aux (bag1, 12 epochs, ~25 s/epoch): day 6 standalone adjusted 0.789126 full /
  0.788991 selection (plain 0.797852); day 5 0.779666 (warm16 on the same cohorts: plain 0.8015 / 0.7946). DIN attention
  0.7740 vs bag 0.7797 on the day-5 validation (overfits from epoch 3); time-gap/repeat/exposure terms -0.002; a previous-
  impressions negative bag -0.006; dense-Adam MAP variant (`nxb_b_mf.py`) <= 0.747. Blend with nxa_h1 (rank 10%, chosen on the
  selection subset; grid in `artifacts/nxb_b_bag1_report.json`): day 6 selection +0.000201, full +0.000242, day 5 +0.000588;
  logistic head instead +0.00011 (selection). **Decision:** below the 0.80 bar and below the +0.0005 day-6 gate -> not queued;
  attention/sequence again generalises worse than the bag. No Kaggle request (fits are 5 min locally; the limit is modelling,
  not compute).


30Oct strongerDAE: user refreshed in-app editor, restoring control. Existing49.861MBbundle uploaded via normalfilechooser to PRIVATEdataset techarena-codex-dae-strong-inputs; no newconsentblock. Imported already-frozen test notebook plus display-only result table. T4GPU selection verified; Save&RunAllVersion1 Fixed DAE stronger control screen accepted; UI RunningwithGPU T4x2. Both daily DAEfits use strict pre-cutoff data, unchanged survivor recipe, no epoch/blend/grid search. No localtraining, holdout/evaluation/submission. Await screen; full only if frozen gate passes.

## 2026-10-30 — nxb round, Track C (coordinator-approved): exact MAP 16-dim choice factors (C1) + joint click/exposure model (C2)

- **Hypothesis (C1):** the rank-16 knee says the simulator is a 16-dim factor model; every current expert is an SGD
  approximation, so an exactly optimised MAP choice model (Newton per user, block Newton per item) should beat them.
  **(C2):** impressions (label-free, ~10x clicks) were chosen by a policy from its belief about the user; a second user vector p
  with a catalogue-softmax exposure likelihood that shares v should sharpen the item factors.
- **Change:** `nxb_c_mf.py` (self-contained torch CPU): event softmax (10 + no-click) with s = u.v + b, s0 = c0 + w_u; optional
  exact catalogue-softmax terms (pre-cut ledger clicks with u; pre-cut impressions with p); Gaussian priors; warm start idf
  click-SVD 16. Phase 1 fits < day 5, best round by held-out day-5 log-loss; phase 2 refits < day 6 (day 5: fit < day 5, round
  re-selected on day 4). Post-processing `nxb_c_post.py` (u shrunk toward kappa*p for 0-4-click events), `nxb_c_laplace.py`
  (posterior-averaged scoring), joint blend `nxb_c_combo.py`. 20 configurations logged in `nxb_journal.md`.
- **Runtime:** C1 ~60 s per round (2 rounds selected); C2 ~240 s per round; every fit < 15 min.
- **Results (adjusted AUC):** events-only C1 0.69 (held-out); + ledger catalogue term 0.760; + prior centred on the SVD factors
  and no-click user offset 0.767 (best C1). Phase 2: C1 day-6 selection 0.776895 / full 0.777413 / day 5 0.767346; C2 0.775655 /
  0.776066 / 0.766056 (exposures make it 0.001 worse; p.v alone 0.572; shrink toward p +0.0001; coupled kappa variant diverges).
  Rank 12 / 16 / 20: 0.7657 / 0.7673 / 0.7633 held-out -> 16 confirmed. References: bag1 0.788991 / 0.789126 / 0.779666;
  warm16 0.791513 / 0.792469 / 0.786732; nxa_h1 0.818331 / 0.818023 / 0.811431; V53 0.817528 / 0.817169 / 0.810911.
  Blends with nxa_h1 (selection-subset grids saved): C1 +0.000229 / +0.000274 / +0.000426; C2 +0.000217 / +0.000258 /
  +0.000402; bag1+C1 (5% each) +0.000289 / +0.000355 / +0.000675 (selection / full day 6 / day 5).
- **Decision:** no survivor (day-6 selection < +0.0005 for every blend); nothing queued. The exact MAP of the u.v choice model
  is worse than early-stopped SGD bag models (0.777 vs 0.789 vs warm16 0.792 on day 6); impressions do not improve the item
  factors. No leakage flags.

## 2026-10-30 — nxb round, last factor-model step: choice-level stacking head (nxb_c_head.py) — REJECTED, nothing queued

- **Hypothesis:** a conditional-logit head (10 candidates + no-click) over logit-ranks of nxa_h1, bag1, C1 and C2 u.v, with a
  no-click utility from history-bin intercepts and event max/mean of each expert, combines the diverse factor experts better
  than rank blends (bag1 + C1 rank blend was +0.000289 selection / +0.000675 day 5). warm16 dropped (evaluation predictions come
  from a different script; day-6 epochs chosen on day 6). Design frozen in nxb_journal.md before any score.
- **Change / runtime:** eval-mix weighted multinomial NLL + L2, 6 lambdas, L-BFGS; fit on day 5 -> day 6, lambda on the day-6
  selection subset (grid artifacts/nxb_c_head_grid.json); day 5 by 2-fold user cross-fit (seed 9901). ~2 min.
- **Results (adjusted):** best lambda 100: day-6 selection 0.817108 (-0.001223 vs nxa_h1), full day 6 0.816607 (-0.001417),
  day 5 0.809585 (-0.001846). Control with nxa_h1 alone through the same head: -0.002381 / -0.003659, so the stacked experts add
  +0.0012 over the control but the within-event renormalisation costs ~0.0024 of global (cross-event) AUC.
- **Decision:** reject; no evaluation predictions, nothing queued. Caveat recorded: bag1/C1/C2 settings were chosen on the day-5
  held-out day. Factor-model family closed for this round (coordinator instruction); next decision with the user.


### Codex stronger-baseline DAE transfer rejected —30October2026

Private KaggleVersion1id363765920, TeslaT4. Both original fixed20epoch DAEfits completed; no hyperparameter/blend search. Fitting/scoring/bootstrap64.523s; totalnotebook106.5s. Same deterministic20%users: day5N7017/day6N4896; frozen H1/V53 primary. Supplied3bin AUC day5 0.816673662->0.816006248(-0.000667414,paired95[-0.000866085,-0.000485384]); day6 0.821542685->0.820965333(-0.000577352,95[-0.000812579,-0.000365666]). Historical5bin: day5 0.814626858->0.813973405(-0.000653452), day6 0.819305255->0.818741797(-0.000563458). Both dates/weightings FAIL+.0002screen; allprimary intervalsbelowzero; NO fullcandidate metrics or confirmation. H1/V39 and unheadedV53 secondary parents also lose. REJECTthisfixedintegration; no tuning/reversal/subgrouprescue. Earlycut3simplified+0.001460 doesNOTtransfer to mainmodel inthisrecipe; no new0.82claim or delta addition to0.818023. Inheritedday5H1same-dayusercrossfit caveat retained; newDAEfits strictlyprecutoff. Downloaded5artifacts, manifest/parents/hashverified, independent sample AUC/predictionformula/cutoff/history/label/order/mass/checkpoint audits pass;29protectedunchanged. No holdout/evaluation/submissions/promotion. Existingmodel andofficialledgerunchanged. Report:results/cdx_dae_strong_report.md. Source/report/checkpoints/receipt atartifacts/v54_horizon/cdx_dae_strong_001/. No GPUorlocaljobsremaining.


### Codex main-tree DAE feature screen prepared, no score yet —1November2026

Fixed proposal: unchanged catalog DAE as10causal scalar inputs inside matched354/364feature main LightGBM trees;735trees, same parameters/rows/seed,20%past events. New factor architectures,AEhyperparameters,seeds,stackers and blend grids not searched. Prefix2..6 rebuilds needed for causal OOFfeatures. Daily allowed first-user cohorts preserved;20%user screen7017/4896,3binand5bin+.0002BOTHdates gate, survivor onlyfull+.0005. ParentH1/V53day5crossfit caveat inherited; not full V53 horizon reconstruction. Announced10–15min Kaggle estimate pending pilot. Preparation was interrupted: original771MBarchive is not validZIP; no notebook import/upload/training/result existed. Repacked identical manifest-verified data, retained failed archive, CRC verified. No evaluation/holdout/submission. Organizer Instructions found at Downloads/Task/INSTRUCTIONS.md; no external-data/shared-code rule specified there. Best reportedlocal0.818023 andofficial0.816623 unchanged. Currentstate records no activejob.

1Nov status verification: Kaggle main-tree draft still contains old copied cell, runtime OFF; no new fit or AUC. Recovered archive1.653GBCRCpassed. Pre-fit code audit corrected Dataset.construct parameter initialization (frozenmax_bin127/minleaf200) via versionedrunner+manifest; no settings change. New runner hash and notebook entrypoint recorded; dataset not uploaded, notebook not imported, no backgroundjob. Best local0.818023/official0.816623 unchanged.
