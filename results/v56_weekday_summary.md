# V56 weekday comparison and sequence-packing finding

The 0.834786 goal remains unmet. Best main-protocol local candidate remains V53: history-adjusted AUC 0.8171690462, ordinary AUC 0.8254233319. Confirmed official best remains V35 = 0.816623. No new CSV was exported or recommended.

## Weekday comparison

Two complete standalone CatBoost classifiers used the same 491,082 fitting rows from days2/3, inverse-probability weights, 354 numerical features and fixed 600-tree settings. The categorical variant added opaque user/item identities and user/item-by-time-bucket identities. IDs were categorical, not interpreted as ordinal numbers. Holdout features received masked clicked labels.

The holdout contains each user's actual first event at/after day4, through training end day7: 47,928 events. Timestamp minima were independently checked. It is 73.1% weekdays versus evaluation's 97.2%; the proxy weights joint history-length/day-type bins. Its three-day horizon is shorter than evaluation's seven-day horizon, and the cohort was previously used in V42 expert comparisons. It is not an untouched holdout or the V53 ensemble, and its absolute scores must not be compared to V53's headline.

| Predictor | Ordinary AUC | Joint history/calendar-adjusted AUC |
|---|---:|---:|
| Numerical control | 0.7965551594 | 0.7972132277 |
| Added categorical identities | 0.7812535004 | 0.7822626521 |

The categorical variant was rejected. Adjusted difference is -0.0149505756; exact paired event bootstrap with 1,000 replicates gives [-0.0158800251, -0.0139827127]. Full saved-model raw-input replay matched every held-out prediction exactly for both models. The fitting-cache join, all listed original/staged input hashes, masked-label path, source repair and memory-workflow provenance passed checks. User-by-bucket identity accounted for 15.86% of the categorical model's feature importance; this alone does not establish why it generalized worse.

## Memory and library limitations

At 98% system RAM use, only the verified V56 process was stopped; its completed numerical checkpoint was preserved. The same fitting matrix was staged in Fortran layout, with all logical values checked, then serialized as a quantized pool. Labels and weights in their original sequence, plus categorical feature indices, were checked after reload. A separate fitting process used a 750mb CTR-cache budget and four logical CPUs. At a later training checkpoint, system RAM was 89% used. This is a measured checkpoint, not a peak-memory guarantee. CatBoost's [performance documentation](https://catboost.ai/docs/en/references/training-parameters/performance) explains that the RAM setting concerns CTR calculation and is not an absolute process-memory cap.

The installed library raised an internal error when asked to predict directly on its quantized categorical pool. A separate 4,096-row synthetic serialization/training probe produced identical raw-input predictions before/after pool serialization, but this does not replace real-data per-object quantized-vs-raw equivalence, which remains unavailable. Full real-data saved-model raw inference was verified separately. Neither probe nor lower fitting loss establishes a generalization gain; [CatBoost's FAQ](https://catboost.ai/docs/en/concepts/faq) also explains why fitting-time and prediction-time training metrics can differ.

This is a recorded resumed experiment. The low-memory workflow expects the prepared V56 feature caches and preserved numerical checkpoint. Large data, pools, models and source snapshots remain under ignored artifacts. Small reports and reproducible source are tracked.

## Additional diagnostics and next action

On fitting days0–4 only, candidate-slot click rates showed no strong unconditional main effect (chi-square p=0.7828; maximum/minimum rate ratio 1.0157). No slot model was fitted. This is not proof that every possible conditional ordering effect is absent. The installed CPU backend also rejected the `ctr_history_unit=Group` capability probe.

A concrete legacy GRU packing bug was reproduced. `_pack_histories` left-padded clicks, while `pack_padded_sequence` consumed only the first `length` positions. Actual clicks were lost for short histories. Tests reproduced corruption at lengths1,2,4,7,10,15; the new `history_sequence.py` helper preserves all clicks with right-padding. Empty histories retain a dummy token solely for the packing API, and future encoders must explicitly mask their output. Nonempty histories shorter than20 account for 59.54% of evaluation users. This concerns the old rejected GRU experiment; no accepted V53 model was changed.

The packing regression passes, but a corrected recurrent model has not yet been trained or scored. Earlier GRU rejections cannot rule out a correctly packed sequence model. [DIEN's primary paper](https://arxiv.org/abs/1809.03672) provides motivation for modeling temporal interests; it is not evidence of a gain on this dataset, and the planned small residual GRU would not be a full DIEN reproduction. The next modeling action is a no-click-aware residual sequence test on unchanged main day6/day5 cohorts. Encode each history once per event, preserve pretrained factors/context, mask empty-history residuals, and pack/profile in small chunks to avoid the large intermediate arrays that previously stressed RAM. Score only after freezing the recipe; no claimed gain follows from the packing fix alone.
