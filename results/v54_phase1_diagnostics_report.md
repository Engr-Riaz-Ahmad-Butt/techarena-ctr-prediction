# Phase 1 diagnostics — no model changes

Scope: all saved selection rows: 352,300 day5 and 244,220 day6. The reserved day6 holdout was not opened or scored. Evaluation was inspected only for authorized candidate/user ID and timestamp counts. No model was trained or changed; no submission was written.

## 1. Drift and recent popularity

No frozen V53 model has saved labeled predictions for successive future days. Day5 is user-OOF and day6 uses a later head, so those AUCs are not day1/day2 drift of the same model. Hourly slices and fixed day5 popularity across days1/2 are available. Days3+ after cutoff require additional labeled dates and saved predictions.

| Diagnostic | Day5 | Day6 selection |
|---|---:|---:|
| V53 AUC | 0.818091 | 0.825527 |
| Impression-popularity AUC last1 | 0.521964 | 0.518530 |
| Impression-popularity AUC last3 | 0.522827 | 0.520886 |
| Impression-popularity AUC last7 | 0.523103 | 0.521470 |
| Impression-popularity AUC all | 0.523103 | 0.521470 |

Last7 equals all-time because only 5/6 prefix days exist. Recent impression popularity does not beat all-time. Fixed day5 popularity, without refreshing counts, gets day2 AUC 0.520113 (last1), 0.521675 (last3), 0.521779 (all). That is a real two-date diagnostic for the same simple score, not for one frozen V53.

V53 six-hour slices after each day-specific cutoff:

| Hours | Day5 | Day6 |
|---|---:|---:|
| 0–6 | 0.825543 | 0.833955 |
| 6–12 | 0.820957 | 0.831278 |
| 12–18 | 0.813985 | 0.823162 |
| 18–24 | 0.810800 | 0.810517 |

This decline can reflect time-of-day and cohort changes, not necessarily model aging. Evaluation prediction-horizon counts are reported below.

## 2. Item age and lifecycle

The catalog has 50,000 items; 99.992% were first observed within24h. First-observation timing within those hours: {'n': 50000, 'mean': 3.466517666666667, 'q10': 0.31666666666666665, 'median': 2.4166666666666665, 'q90': 7.933333333333334}. This is age within the available window, not actual catalog creation age.

| Quantity | Day5 clicked / not-clicked mean | Day6 clicked / not-clicked mean |
|---|---:|---:|
| age_days | 5.37098258 / 5.36541656 | 6.37238655 / 6.36661645 |
| recent3_to_all_popularity_share_ratio | 0.999421947 / 1.0001696 | 0.999675298 / 1.00031399 |
| alltime_impression_share | 3.24212631e-05 / 3.07986994e-05 | 3.22687362e-05 / 3.07396324e-05 |

Older-first-seen score on ALL users has AUC 0.505028/0.505347. On full zero-history cohorts it is 0.511633/0.573303. The earlier promising small-cohort age result does not generalize into a strong all-user signal. Age within each event ranks the clicked candidate only slightly above random (see table). Recent/all popularity-ratio AUC is 0.497492/0.497877.

## 3. Candidate generation

Rank1 = highest score/most popular/oldest. Only events with a click enter this distribution. Ties share mass across tied ranks; a flat reference is 10% in each rank. Flat clicked ranks are not implied by uniform negative sampling: clicked items may have different relevance.

| Day / score | Rank1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day5 alltime_popularity | 10.77% | 10.97% | 10.72% | 10.52% | 9.94% | 10.21% | 10.10% | 9.34% | 8.93% | 8.49% | 5.287 |
| Day5 last3_popularity | 10.77% | 10.96% | 10.59% | 10.40% | 10.13% | 10.20% | 9.97% | 9.40% | 9.17% | 8.43% | 5.296 |
| Day5 older_item_age | 10.13% | 10.09% | 10.37% | 10.10% | 10.14% | 10.31% | 9.96% | 9.93% | 9.56% | 9.41% | 5.437 |
| Day5 ALS | 31.99% | 18.85% | 13.11% | 10.53% | 7.66% | 6.13% | 4.74% | 3.35% | 2.42% | 1.21% | 3.201 |
| Day6 alltime_popularity | 10.95% | 10.27% | 10.46% | 11.42% | 10.05% | 9.82% | 9.70% | 9.22% | 9.38% | 8.74% | 5.311 |
| Day6 last3_popularity | 10.91% | 10.30% | 10.58% | 10.96% | 10.04% | 10.19% | 9.78% | 8.85% | 9.48% | 8.90% | 5.320 |
| Day6 older_item_age | 10.38% | 10.31% | 9.94% | 10.37% | 10.23% | 10.68% | 9.65% | 10.04% | 9.25% | 9.14% | 5.401 |
| Day6 ALS | 32.19% | 19.04% | 13.25% | 10.42% | 7.62% | 5.91% | 4.48% | 3.54% | 2.30% | 1.25% | 3.182 |

Negative candidates concentrate in the highest popularity decile: 27.08%/27.06%, versus catalog shares 10.05%/10.07%. Bottom-decile negative shares are 3.85%/3.94%. Their distribution across popularity deciles is much closer to historical impression-weighted sampling (total variation 0.0089/0.0090) than uniform catalog sampling (0.2318/0.2305). This supports popularity-skewed exposure, not a proven negative-sampler algorithm. No useful recent-popularity tilt was detected. Clicked items are mildly more popular than negatives; ALS separates them much more strongly.

## 4. Cold items

Validation day5: 0/352,300 unseen and 0 first-seen-within-two-days. Day6: 0/244,220 for both. Evaluation: 0/529,390 for both. Their V53 AUC is undefined because there are no rows. Cold-item handling cannot explain the observed gap within this catalog/window.

## 5. User-level structure, positions, IDs

Across 686,824 training events and 53,003 users: 3.60% of consecutive same-user events share any candidate; mean overlap 0.03635/10; zero identical consecutive slates; 2.56% of presentations had appeared in an earlier event for that user. These structure counts inspect candidate IDs/timestamps, not labels.

Day5 clicked-position percentages, positions1–10: 10.35, 10.19, 9.87, 10.26, 10.08, 9.66, 9.67, 9.82, 9.87, 10.23. Uniformity chi-square p=0.1646.
Day6 clicked-position percentages, positions1–10: 10.32, 9.84, 9.53, 10.09, 9.75, 10.46, 9.79, 10.53, 9.95, 9.73. Uniformity chi-square p=0.0685.

Raw user/item IDs are hash-like strings, not natural numeric ranges. We tested their hash-order deciles, separately from insertion-order encoded IDs. Encoded indices partly encode first appearance and should not be mistaken for meaningful identifier ranges.

Day5 users: raw-hash correlation with click=+0.000520, timestamp=+0.007162; encoded-index correlation with click=+0.001368, timestamp=+0.205556; raw-hash decile click rates 6.249%–6.417%.
Day5 items: raw-hash correlation with click=-0.000557, timestamp=-0.000856; encoded-index correlation with click=-0.010630, timestamp=-0.000687; raw-hash decile click rates 6.197%–6.456%.
Day6 users: raw-hash correlation with click=-0.001989, timestamp=+0.002140; encoded-index correlation with click=-0.000155, timestamp=+0.212511; raw-hash decile click rates 6.212%–6.503%.
Day6 items: raw-hash correlation with click=+0.000379, timestamp=+0.001022; encoded-index correlation with click=-0.009775, timestamp=+0.001417; raw-hash decile click rates 6.123%–6.535%.

## 6. Simple-score ceiling and residual correlations

These are observed AUCs of simple scores, not a mathematical ceiling for future models. V53 is an ensemble rank score, not a calibrated probability. Correlation with label minus that rank can reflect score scaling; it does not measure independent predictive gain.

| Signal | Day5 AUC / residual Pearson r | Day6 AUC / residual Pearson r |
|---|---:|---:|
| pop_all | 0.523103 / +0.016601 | 0.521470 / +0.013074 |
| pop_last3 | 0.522827 / +0.016486 | 0.520886 / +0.013230 |
| ALS | 0.757775 / -0.489916 | 0.760686 / -0.487990 |
| item_age_older_higher | 0.505028 / -0.015636 | 0.505347 / -0.018103 |
| recent3_all_share_ratio | 0.497492 / -0.002310 | 0.497877 / +0.000568 |

ALS is the single signal most correlated in absolute value with the requested rank residual: r=-0.489916/-0.487990. This mostly reflects shared ranking structure and the mismatch between ranks and probabilities; it is not evidence that another ALS blend would help. Existing model/stacker work remains closed.

## 7. Evaluation user activity and prediction horizon

Training activity means observed impressions, not warm-up clicks. Gap is cutoff minus last training activity among previously seen users; users with no earlier impression are listed separately.

| Cohort | No prior activity | Median cutoff gap, days | 90th percentile | Within1day | Within2days |
|---|---:|---:|---:|---:|---:|
| Day5 | 873/35230 | 0.2375 | 1.7403 | 74.69% | 88.58% |
| Day6 | 399/24422 | 0.2354 | 1.8626 | 75.31% | 88.94% |
| Evaluation | 1453/52939 | 0.3583 | 2.5983 | 65.60% | 81.69% |

validation5: events by day1/day2/… after cutoff={'1': 35230}; median prediction horizon=0.4688 days; median time since last training activity at actual event=0.8146 days.

validation6: events by day1/day2/… after cutoff={'1': 24422}; median prediction horizon=0.4736 days; median time since last training activity at actual event=0.8174 days.

evaluation: events by day1/day2/… after cutoff={'1': 35269, '2': 8764, '3': 3914, '4': 2236, '5': 1273, '6': 898, '7': 585}; median prediction horizon=0.7021 days; median time since last training activity at actual event=1.3132 days.

Evaluation users are less recently active at the cutoff, and evaluation has a longer prediction horizon than the one-day selection windows. This is a real descriptive mismatch that history-length reweighting alone does not remove.

## Ranked follow-up hypotheses — approval required before any modeling

1. Prediction-horizon and user-activity freshness mismatch: evaluation spans seven post-cutoff days and has older user activity; quantify matched fixed-model future-date behavior if later labeled dates exist. This is a diagnostic lead, not permission to reopen short-history, bucket or weighting recipes.
2. Candidate-generator popularity skew: negative candidates are far from uniform over items and match historical exposure weights closely. Obtain the organizer sampling specification before deciding if it leaves any unmodeled information. Existing popularity/stacker models are not reopened.
3. ALS/personal preference structure: standalone ALS AUC is about0.76 and clicked items lead its rank distribution strongly. It is already in V53 and previous diversity tests failed, so incremental value is unproven. It ranks third only as a structural clue; there are not three strong new residual signals in these diagnostics.

Needed: labeled predictions from the SAME frozen V53 across later dates for a genuine multi-day drift curve, and organizer information on how the nine negatives are generated. No new models or submissions should be built without user approval.

