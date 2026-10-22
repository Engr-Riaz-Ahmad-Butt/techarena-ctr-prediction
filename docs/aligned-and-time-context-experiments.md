# Aligned warm factors and timestamp context

V18's user-confirmed official AUC is0.809189. The leaderboard screenshot supplied on2026-10-22 placed Orbit16th; scores and positions can change. Local validation remains separate from official scoring.

## Aligned coordinate interactions

The previous classifier received scalar preference similarities. The new variant adds32 coordinate products, exposing which parts of user and history preference agree with each candidate. To make coordinates comparable across chronological blocks, each16-dimensional warm-factor model is rotated to a fixed pre-block2 reference using orthogonal Procrustes on normalized item vectors. The same orthogonal rotation applies to U,V,H. Each scored block still uses a checkpoint fitted strictly before that block. Scalar predictions are invariant under the rotation.

The classifier receives258features: the existing226 plus16 normalized user/item products and16 normalized history/item products. Settings remain599trees, depth7, learning rate0.08, seed42. January26 selects0%,50%,100% replacement of the V18 decomposition component using history-weighted AUC; all other ensemble weights are fixed. Full replacement was selected and frozen for January25.

| Validation | V18 AUC | Aligned AUC | V18 history-weighted AUC | Aligned history-weighted AUC |
|---|---:|---:|---:|---:|
| January26 | 0.816800294 | 0.817948106 | 0.808540088 | 0.809611651 |
| January25 | 0.808296773 | 0.810644716 | 0.801149062 | 0.803552134 |

Independent replay verifies both dates, exact blend reconstruction, event/label alignment, probability validity, rotation optimality, sampled coordinate features and preservation of cosine similarities. Both ordinary and weighted conditional paired-event improvement intervals are positive. These dates have been reused repeatedly; the intervals do not account for selection bias or public/private leaderboard distribution differences. The local0.82 threshold has not been reached by this variant.

## Timestamp context experiment

A second hypothesis adds32 timestamp/activity features to V18's226features: elapsed click ages, activity span, time-bucket/day-type history composition, and four H-factor history profiles conditioned on recency or matching time of day. All histories must precede their event, and all checkpoint labels must precede their training block. A10000-row smoke check passed; bucket extraction was checked against every encoded training and evaluation event.

The original compute-saving gate would have skipped this arm because the aligned ordinary gain exceeded0.001. The scope was explicitly expanded after aligned day6 and before any timestamp-context AUC, because the competition gap remains substantial. This deviation is recorded in [the timestamp plan](time-context-plan.md). If the aligned candidate confirms independently, the timestamp variant must beat its frozen day6 weighted AUC0.809611651 before a confirmation fit. No timestamp result has yet been used to change this floor or model settings.

## Reproduction after V18

```powershell
python imp5_aligned.py
python imp5_test_aligned.py --tag aligned
python imp5_timecontext.py
python imp5_test_aligned.py --tag timecontext
```

Select a variant using January26 weighted AUC among independently confirmed candidates. Only then run `imp5_finalize.py --tag <selected> --version 19`, followed by `imp5_test_final.py --tag <selected> --version 19`. Generated CSVs, models and arrays are excluded from Git. V18 is preserved. No official submission is performed by these scripts.

See [the aligned plan](aligned-factor-plan.md) for the original protocol. Metric reports are stored in `results/`; experiment artifacts are stored outside Git.
