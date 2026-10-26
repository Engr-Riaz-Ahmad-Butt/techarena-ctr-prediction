# Progress toward requested score 0.834786

The target remains unmet. The last user-confirmed official score is V35 = 0.816623. The user confirms V36 and V39 were not uploaded and wants local history-adjusted AUC at least 0.82 before another upload recommendation. No new submission CSV was exported in this work.

V46 jointly fine tunes warm factor embeddings, recent-history transforms and item/calendar biases on fitting labels only. Its best adjusted AUC was 0.8164093032, below V39's 0.8166614879. It was rejected. Independent initialization, saved-model replay, causal cutoff and metric checks passed. Falling fitting loss from 0.9309 to 0.7096 did not establish validation improvement.

V47 adds strictly pre-cutoff supplied histories of unseen users to a matched ALS continuation. Its day6 support was only 503 auxiliary users and 1,764 clicks. Treatment adjusted AUC was 0.8166951927, versus 0.8166942396 for equal-length control continuation. Both gains were uncertain, and the extra-history effect was not distinguishable from zero. It was rejected. Saved-model and metric replays passed for both arms. An implementation repair restored explicit zero target entries after sparse addition; the original source, frozen settings and repair manifest are retained locally. No tuning accompanied the repair.

V48 uses a regularized conditional logistic calibrator with fixed spline, history-bin and event-context features. It fits only a predetermined random half of day5 users using causal day5 parent predictions. It is checked on the other half of day5 users and then on all day6 users, without changing its settings.

| Validation | Baseline ordinary | V48 ordinary | Baseline history-adjusted | V48 history-adjusted |
|---|---:|---:|---:|---:|
| Day5 held-out users only | 0.8175247395 | 0.8176195825 | 0.8108704049 | 0.8109907092 |
| Day6, future date | 0.8249578084 | 0.8250648202 | 0.8166614879 | 0.8168150072 |

Independent direct logistic replay, source/input hashes, user partition, full recipe and metrics passed. Exact paired event bootstrap with 1,000 replicates per split gave adjusted-gain intervals of [0.0000025621, 0.0002339894] on held-out day5 users and [0.0000575935, 0.0002394324] on day6.

**Protocol limitation:** day5 is a same-date user holdout for the calibrator, not a second strictly future temporal fold. Its AUC must not be compared directly with older full-day5 headline scores. Day6 is later than all calibrator fitting labels. Both parent dates were repeatedly reused previously; the intervals condition on the chosen recipe and do not remove selection uncertainty or weekday/evaluation distribution shift. These are local diagnostics, not official-score forecasts.

The gain supports further investigation of conditional cross-event score calibration rather than more ALS capacity/continuation sweeps. A translated full-data calibrator and evaluation replay have not been produced; V39 remains the latest exported CSV. The active target stays 0.834786, and a small local gain is not completion.
