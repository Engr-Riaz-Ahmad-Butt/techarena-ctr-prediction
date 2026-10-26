# Expert calibration progress toward 0.834786

Previous goal turn classified as progress: V48 produced a saved, replayed calibrator with positive conditional bootstrap intervals. This turn extends that approach rather than repeating ALS capacity sweeps.

The target 0.834786 remains unmet. The official confirmed score remains V35 = 0.816623; no newer upload or official score is claimed. No new CSV was exported, preserving the user's minimum local adjusted-AUC requirement of 0.82 before another upload recommendation.

V49 learns a fixed logistic combiner from 161 causal features, including existing tree, neural, frozen-context and ALS experts. It uses V48's fixed half-user training partition on day5. Held-out day5 adjusted AUC rises from 0.8109907092 to 0.8111278453, but the gain interval includes zero. It is rejected, with no future-date fit/selection or promotion.

V50 retains V49's feature recipe, C=0.01 and 25% blend, but uses complementary user folds. Each day5 user receives a prediction from a calibrator that did not see that user's labels. After that crossfit gate passes, a fresh calibrator fits all day5 causal scores/labels and predicts the later day6 cohort. No hyperparameters or blend weights are tuned on either validation result.

| Cohort | Baseline | Baseline ordinary | V50 ordinary | Baseline adjusted | V50 adjusted |
|---|---|---:|---:|---:|---:|
| Full day5, user-crossfit | V39 | 0.8176714756 | 0.8179616247 | 0.8105265996 | 0.8107894042 |
| Day6, future date | V48 | 0.8250648202 | 0.8252452244 | 0.8168150072 | 0.8169941524 |

Source/input hashes, exact user partition, fitting-label chronology for day6, direct logistic predictor replay and full rank recipes passed independent checks. Saved-score replay error is zero. Exact paired event bootstrap, 1,000 replicates per cohort, gives adjusted-gain intervals [0.0001364128, 0.0003804437] on day5 versus V39 and [0.0000864036, 0.0002764580] on day6 versus V48.

Day5 is a user-OOF check rather than a second strictly future temporal fold. Its full-cohort score cannot be directly compared to V48's half-cohort day5 headline. Both dates were repeatedly reused during earlier modeling. Bootstrap conditions on trained calibrators and excludes fitting/selection uncertainty; evaluation is predominantly weekdays while these cohorts are weekends. No official-score forecast follows from the local gain.

V50 is the latest local candidate. V39 remains the latest exported submission. A translated final-fit calibrator and evaluation prediction replay remain unimplemented; the 0.834786 objective stays active. Further changes should investigate new predictive signals, including real expert probability/event-mass information and causal propensity summaries, rather than merely increasing blend search volume.
