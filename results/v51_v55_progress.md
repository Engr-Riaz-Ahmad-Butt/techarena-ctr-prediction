# V51–V55 progress toward 0.834786

This wave supplies new modeling evidence and a higher audited candidate, V53. The requested score remains unmet, and no new CSV or official score is claimed.

V51 added actual expert probabilities, event mass and conditional entropy to the existing rank inputs. Day5 user-crossfit improved slightly, but future day6 adjusted AUC was 0.8169924240 versus V50's 0.8169941524. It was rejected; saved predictor and metric replay passed.

V52 tested matched conditional-choice warm models with 16 and 32 dimensions. Both received identical negative samples, event ordering, learning rate, regularization and additional epochs. The best expanded-arm primary adjusted AUC was 0.8170046288, but its gain interval included zero against both V50 and the matched 16-dimensional control. It was rejected. Independent NumPy scoring, cutoff and rank/metric checks passed for both selected models; raw replay errors were below 1.3e-7.

V53 adds strictly pre-cutoff user/item propensity and time-bucket rates, one/three-day activity summaries, and supplied history age/context statistics. Settings were fixed before validation. It keeps V50's two user folds on day5 and fits a full-day5 calibrator only for future day6 predictions.

| Cohort | V50 ordinary | V53 ordinary | V50 adjusted | V53 adjusted |
|---|---:|---:|---:|---:|
| Full day5, user-crossfit | 0.8179616247 | 0.8180913841 | 0.8107894042 | 0.8109113853 |
| Day6, future date | 0.8252452244 | 0.8254233319 | 0.8169941524 | 0.8171690462 |

Independent saved-model replay is exact. Counterfactual feature tests rotated 247,048 post-cutoff click labels without changing a new feature; changing a prior fitting label did change features, and a user-event-count feature matched a raw-record oracle. Source files were not modified. Exact paired event bootstrap, 1,000 replicates per cohort, gave adjusted-gain intervals [0.0000808083, 0.0001631800] on day5 and [0.0001353959, 0.0002141837] on day6, both versus V50.

V54 decomposed prediction into event-click probability and conditional item choice. The softmax gradient matched finite differences with error 4.2e-11. Its day5 gain was uncertain, so no future fit was started. Direct saved-predictor replay and probability/partition checks passed; rejected.

V55 fitted 100 shallow residual trees above fixed V53 logits, with parent and residual fitted to the same training-user fold. Its day5 adjusted AUC fell to 0.8108794197 versus V53's 0.8109113853; no future fit was started. Reloaded tree replay was exact; rejected.

**Limits:** Day5 is user-OOF rather than a second strictly future temporal fold. Both dates have been repeatedly reused and are weekends, while evaluation is predominantly weekdays. Bootstrap conditions on fitted models and does not correct fitting, repeated-selection or calendar-shift uncertainty. Official best remains V35 = 0.816623. V39 remains the latest exported CSV; V53 is a local candidate only. The user's minimum local adjusted AUC before another upload recommendation is 0.82; the goal is still 0.834786.

The next substantial validation task is a predeclared first-future-event weekday fold with a causal full-model baseline. Component AUC must not be presented as a submission score. These results give little support for more small capacity/continuation or nonlinear calibration sweeps alone closing the gap.
