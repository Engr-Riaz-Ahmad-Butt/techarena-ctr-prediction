# Numeric neural choice experiments

These experiments train on three strictly earlier date blocks of the prepared
354-feature causal cache. Entire 10-candidate events remain together. An
11-option softmax includes no-click; predictions cover only the ten candidates.
All normalization uses fitting rows. Training uses fixed epochs and history-bin
importance weights derived from the unlabeled evaluation population. All trials
use a fixed 10% expert / 90% V53 rank blend. Validation labels enter scoring only.

| Trial | Change | Day6 adjusted | Day5 adjusted | Outcome |
|---|---|---:|---:|---|
| 001 | Numeric MLP, symmetric mean/max candidate pooling, 8 epochs | 0.817390213 | 0.810623039 | Confirmation failed |
| 002 | Mask fitting-constant features | 0.817390213 | 0.810676244 | Confirmation failed |
| 003 | Standardize varying features within each event; retain shared event features | 0.817295085 | 0.811162140 | Both date gates passed |
| 004 | Same setwise network, fixed 32 epochs | 0.816923969 | Not run | Day6 failed |

Every completed trial reloaded its saved checkpoint and reproduced every raw
prediction exactly. Trial003 improves over V53 by 0.000126039 on day6, with
conditional event-influence interval [0.000027906, 0.000224173], and 0.000250754
on day5, interval [0.000148969, 0.000352539]. Its day6 point estimate is slightly
higher than the prior verified XGBoost blend (0.817263437). It remains below the
0.82 upload recommendation threshold and the requested 0.83 target. No CSV export.

The label-free drift diagnostic samples 60,000 fitting and 20,000 validation
rows for each cutoff. It identifies the unsupported day-type column on day5
and a 2.947 fitting-standard-deviation shift in warm-context column283, versus
0.539 on day6. These are descriptive measurements, not proof that covariate
drift caused the scoring difference. Relative normalization is tested through
candidate permutation symmetry and per-event positive affine invariance.

Trial004 reduced fitting loss but worsened held-out AUC, so longer training was
rejected. This is direct evidence against treating falling training loss as
proof of competition improvement.

A subsequent fixed 80% V53 / 10% reproduced XGBoost / 10% trial003 ensemble is
declared in tabular_choice_plan.json. It must preserve row alignment and replay
both models, and improve against V53 and both individual expert blends on both
dates. No blend grid is evaluated. Its result is recorded separately.

All dates have been repeatedly reused. Confidence intervals condition on the
fitted predictions and omit model-fitting and repeated-selection uncertainty.
They are development evidence, not independent confirmation or official-score
forecasts. The last confirmed competition result remains V35 = 0.816623.

## Fixed ensemble result

The reproduced XGBoost metrics exactly match the prior main003 results. The fixed80/10/10 ensemble scores 0.8173193042301217 adjusted on day6 and 0.8112273931434458 on day5. Reloading both saved experts reproduces all predictions exactly. It improves versus V53, but gain intervals versus the neural-only blend cross zero on both dates; the day6 interval versus the tree-only blend also crosses zero. It is not promoted under its declared stricter rule. The best eligible candidate remains neural setwise003: adjusted0.817295085401266, ordinary0.8255597973402244. No0.83 target attainment or new official result.

Kaggle Version6 (Numeric choice and fixed ensemble results) was saved with outputs. The UI explicitly confirmed Quick version complete. The results ZIP includes both neural checkpoints and their recipes in addition to the reproduced XGBoost checkpoints and prediction files. Code and local reports are committed under Riaz Ahmad Butt. No fitting remains active.
