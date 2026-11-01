# History-matched validation and distribution experiments: final report

Stop: no candidate passed +0.0005 history-reweighted AUC on both days. Retained cumulative gain from steps 2–4 is zero, below the requested +0.001 threshold. Keep V53. No holdout evaluation or submission generation.

## Fairer baseline

Weights = target history share / observed cohort history share. Target: 0 clicks 1.26%, 1–5 clicks 16.37%, 6–20 clicks 82.37%. Applied as ROC-AUC sample weights, not an average of separate segment AUCs. These are supplied marginal-history weights; they do not fully replicate evaluation user/item/calendar distribution.

| Cohort | Day 5 unweighted / reweighted AUC | Day 6 unweighted / reweighted AUC |
|---|---:|---:|
| Full selection | 0.818091384 / 0.813119135 | 0.825526709 / 0.820037962 |
| 20% sample | 0.820437259 / 0.816182114 | 0.825858710 / 0.821453597 |

Step2 refits only V53’s original final logistic model: unchanged 190 features, C=.01, scaler, and .75 parent/.25 head rank blend. All upstream experts remain fixed. This is not a new stacker architecture or full upstream pipeline retrain. Weight1 matches the target; weight2/4 multiply short-history training weights further. Weight0 is an unweighted matched retrain control. Only early40% of sampled day5 fits this head.

Steps3/4 train on first-user events from days3/4 (20% hash sample). Each training day’s counts/profile/ALS checkpoint precedes its own cutoff. Step3 uses every earlier training-window click in sparse user-item counts and item-factor profiles, Bayesian priors 5/20, smoothed time_bucket×day_type item popularity/CTR, and ALS features; fits only short histories. Step4 adds user bucket affinity/support/CTR and blends either bucket3 only or all buckets.

All comparisons below use later60% of sampled day5 (42,140 rows) and sampled day6 (48,960 rows), excluding the reserved holdout. The day5 restriction avoids fitting any new predictor on its tested labels. Its baseline differs from the full sample. The early-day5 monotone probability map used for blends was approved in the prior batch and is reused as a protocol, not fitted on test labels.

| Best shared setting | Day5 weighted gain | Day6 weighted gain | Day5 unweighted gain | Day6 unweighted gain |
|---|---:|---:|---:|---:|
| Step2: original_V53_final_head_weight1 | -0.000378910 | +0.000019142 | -0.000382951 | -0.000074395 |
| Step3: short_only_w0.05 | -0.000016408 | -0.000104270 | -0.000012853 | -0.000061942 |
| Step4: bucket3_only_w0.05 | -0.000129366 | -0.000078198 | -0.000140192 | -0.000091216 |

No configuration passes the gate. No rejected change was promoted or combined into the baseline.

## Zero-history probes

| Feature | Day5 AUC | Day6 AUC |
|---|---:|---:|
| item_popularity_by_bucket | 0.563389 | 0.587917 |
| item_popularity_by_bucket_daytype | 0.500000 | 0.570960 |
| joint_item_CTR | 0.602604 | 0.520661 |
| first_appearance_age_older_higher | 0.537270 | 0.627672 |
| first_appearance_recency_newer_higher | 0.462730 | 0.372328 |
| ALS_item_bias | 0.593923 | 0.626959 |

There are only 17 positive rows in the day5 cold cohort and 11 in day6. Several item-level signals score above 0.5 descriptively, but this is weak evidence: the event-bootstrap intervals are stored in artifacts/v54_distribution_zero_probes.json. Neither those signals nor the dedicated short model produce a passing global weighted gain. First-appearance age and recency have opposite score directions; both are explicitly reported.

## Checks and limits

All 32 saved prediction arrays were checked for finite values, shape and exact metric reproduction. Weight sums reproduce the requested mix. Full earlier-window click counts spot-checked directly against training rows. Prefix timestamps and ALS cutoffs asserted. No evaluation file was needed in this batch. V53/V20, baseline_backup and submission artifacts were only read where needed, never written. The separate holdout was not opened or scored. No gain > +0.01 occurred.

Inherited limitation: V53 day5 parent scores/features include the original user-cross-fitted meta pipeline, which is out-of-sample by user rather than a fully temporal parent model. All new feature statistics and dedicated model fitting use earlier data. These heavily reused dates do not provide fresh independent generalization evidence.

The sample results do not support the +0.005 official improvement goal. They show a plateau for the tested recipes, not a mathematical ceiling or proof a better model is impossible. Matching history mix lowers the baseline relative to unweighted AUC; that is a fairer descriptive comparison, not a new official leaderboard estimate. No positive leaderboard gain can be claimed from this batch.

Reproduce from the project folder: `python v54_distribution.py 1`, `python v54_distribution.py 2`, `python v54_distribution.py 3`, `python v54_distribution.py 4`, then `python report_v54_distribution.py`. Existing source models, original feature scripts and experiment2 sample caches are required. New caches/predictions use only the v54_distribution prefix.
