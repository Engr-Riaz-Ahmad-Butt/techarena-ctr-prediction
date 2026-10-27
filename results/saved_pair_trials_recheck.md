# Existing pair-feature trials: independent prediction recheck

No model was trained. Existing saved arrays were read without modification.
All six date/trial combinations have identical events, labels and final V53
comparator arrays. The fixed 80% V53 / 20% candidate rank blends reproduced
exactly. Ordinary and evaluation-history-weighted AUCs match their saved reports.

| Existing trial | Day5 adjusted AUC | Day6 adjusted AUC | Both conditional gates |
|---|---:|---:|---|
| V65 pair propensity | 0.810993351 | 0.817277270 | pass |
| V66 decayed pair propensity | 0.811017391 | 0.817276974 | pass |
| V67 event pairwise ranker | 0.810160880 | 0.816819455 | fail |

V53 comparators are 0.810911385 and 0.817169046. The stronger V65/V66 point
estimates exceed the recent Kaggle interaction candidate's day6 adjusted
0.817263437 by only about 0.000014. They remain exploratory: this recheck does
not reproduce feature construction, saved model inference, or crossfit fitting.
It cannot promote either as the new audited best, certify an official score,
or establish the 0.83 objective. Conditional intervals also exclude repeated
model selection uncertainty.

The V67 ranker declines on both dates, with both conditional gain intervals
entirely below zero. This reinforces avoiding another repeat of the tested
within-event pairwise recipe, which did not improve global AUC.

The prepared regularized neural GPU experiment remains the next queued action.
The prior browser approval rejection is unchanged; the user requested resuming
after normal access recovers. No browser workaround or indirect Kaggle action
was attempted. The hourly follow-up remains configured.

Detailed metrics, conditional intervals, saved-prediction hashes and comparator
hashes: `saved_pair_trials_recheck.json`. Source: `audit_saved_pair_trials.py`.
