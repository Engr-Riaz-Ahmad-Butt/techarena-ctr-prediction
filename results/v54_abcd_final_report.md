# V54 A–D final research report

No candidate passed the +0.0005 global-AUC gate on both selection samples. Retained cumulative gain is zero, below the requested +0.001 stopping threshold. Stop this research batch and retain V53. No new submission is justified.

## Segment shares

These are full selection cohorts (day 6 excludes the reserved holdout). Evaluation has 529,390 rows; only descriptive item/history fields were used.

| Segment | Day 5 count / share | Day 6 count / share | Evaluation count / share |
|---|---:|---:|---:|
| History 0 | 1,970 / 0.56% | 1,340 / 0.55% | 6,650 / 1.26% |
| History 1–5 | 36,430 / 10.34% | 22,130 / 9.06% | 86,680 / 16.37% |
| History 6–20 | 313,900 / 89.10% | 220,750 / 90.39% | 436,060 / 82.37% |
| Repeat | 8,990 / 2.55% | 6,234 / 2.55% | 13,467 / 2.54% |
| Non-repeat | 343,310 / 97.45% | 237,986 / 97.45% | 515,923 / 97.46% |

Perfectly resolving every zero-history positive/negative, including all comparisons against other segments, raises full-selection AUC at most by +0.002292 / +0.002237 on days 5/6. This is a label-informed oracle bound, not a usable feature or expected model gain. Even perfection on those rows does not supply +0.005. Evaluation gain cannot be computed without labels.

## Model screens

All models fit the earliest 40% of sampled day-5 events (28,030 rows). Day-5 results below use only the later 42,140 rows; day 6 uses 48,960 sampled selection rows. This earlier-block protocol was explicitly approved by the user. Scores cannot be compared to previous full-day sample headlines without matching these cohorts.

| Best shared configuration in each family | Day 5 baseline → candidate | Day 6 baseline → candidate |
|---|---:|---:|
| B: shallow_lightgbm_w0.1 | 0.812626070 → 0.812646380 (+0.000020310) | 0.825858710 → 0.825828799 (-0.000029911) |
| C: repeat_only_w0.1 | 0.812626070 → 0.812637741 (+0.000011671) | 0.825858710 → 0.825836094 (-0.000022616) |
| D: new_three_stacker_seeds_w0.05 | 0.812626070 → 0.812640324 (+0.000014254) | 0.825858710 → 0.825843902 (-0.000014808) |

Per-segment isotonic calibration also failed: best shared choice segment_isotonic_w0.1, gains -0.000037887 / +0.000000090.

The dedicated repeat model had only 718 early-block repeat training rows. At its best shared 10% blend, within-repeat AUC fell from 0.683534 to 0.680968 / 0.668188 to 0.665637 (days 5/6). Every non-repeat score was preserved exactly.

D reused the three saved neural seeds, a distinct existing tree model, and an ALS model, then also fitted three new shallow stacker seeds. None passed. These saved components are already ingredients of V53, so their marginal diversity is limited. Full V53 base pipelines were not retrained with new seeds; the original logistic meta learner is deterministic. The new three-seed experiment is second-stage diversity, not three full V53 replicas.

## Leakage and reproducibility

Fit timestamps are strictly below 1737796080; all reported day-5/day-6 evaluation timestamps are later. Repeat features/statistics retain the prior experiment’s strict pre-cutoff construction and audited history timestamps. Saved model source rows were joined by event and candidate position, with exact label alignment assertions. No evaluation labels or evaluation-derived model statistics were used; evaluation access was limited to the authorized segment counts. The reserved holdout was never opened in this batch. No V53, V20, baseline backup, or submission file was written.

Inherited limitation: V53 day-5 scores are user-cross-fitted out-of-sample scores, not fully temporal parent predictions; the original V53 meta layer uses other day-5 users. The new models themselves use only the approved earlier time block. Selection days and model components have been heavily reused, so these results are screens rather than untouched performance estimates.

All 64 saved prediction arrays were checked for shape, finiteness, and exact metric reproduction. No gain exceeded +0.01. The full-cohort and holdout stages were not warranted because every sample gate failed.

Reproduce from the project folder with `python v54_abcd.py A`, then `python v54_abcd.py B`, then `python v54_abcd.py C`, then `python v54_abcd.py D`, then `python report_v54_abcd.py`. A repeats only the explicitly authorized evaluation segment count. Existing experiment-2 caches and model prediction artifacts are required.

## Honest ceiling assessment

These tests support a practical plateau for the current features and model family near existing V53 performance. They do not establish a mathematical ceiling or prove the leaderboard target is impossible. The +0.005 target is unsupported by the measured residual signals; no positive leaderboard gain should be claimed from this batch. Global validation AUC and historical history-adjusted AUC are different quantities; the higher global values do not imply an official improvement. Keep the previously validated submission; no new upload file was created.
