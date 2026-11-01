# V61 weekday-causal benchmark: rejected

**Decision: reject; no supported candidate score and no promotion.** The stored day5/day6 V61 AUCs (0.833840 / 0.842950) are training-set scores. The script calls `fit(x_aug, z['y'])` and then predicts the same `x_aug`; held-out labels therefore train the model whose score is reported. The report calls these values a probe and states that a strict benchmark is still required.

The apparent baseline is also mislabeled: `v53_auc` is calculated from `z['pred']`, not the final V53 predictions. Recomputed against the aligned saved cohorts, final V53 is 0.818091 ordinary / 0.810911 history-adjusted on day5 and 0.825423 / 0.817169 on day6. V61's event IDs and labels do match V53's, but its stored baseline does not. Thus the apparent V61 improvements cannot be used as evidence.

The appropriate comparison protocol remains V53's full-day5 user-crossfit gate followed by fitting on day5 and evaluation on future day6. V53's adjusted gains over V50 were +0.000121981 (day5; paired-event interval [0.000080808, 0.000163180]) and +0.000174894 (day6; [0.000135396, 0.000214184]). V61 was not evaluated under this protocol. The validation dates have also been repeatedly reused, so even a corrected rerun must disclose that limitation; they are not untouched confirmation data.

The extra context aggregates are cutoff-restricted, but this cannot repair same-cohort target leakage. No frozen V61 manifest or independent out-of-sample audit exists. V53 remains the best accepted local candidate at adjusted day6 AUC 0.817169; the 0.82 upload threshold and 0.834786 target remain unmet. No official score is inferred.

Decision details and reproducible reference metrics: `v61_weekday_causal_decision.json`.
