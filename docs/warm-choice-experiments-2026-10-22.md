# Auxiliary click training: 2026-10-22

## Selected model

The new objective combines observed ten-candidate event choices (including no-click) with sampled-negative training on each user's earliest historical clicks. All auxiliary clicks precede the training cutoff. Five identical user/item/timestamp records are deduplicated; repeats at different times remain valid observations.

The selected model has 16 factors and auxiliary weight 1.0. A fixed smoothed item-frequency logit offset in the auxiliary loss accounts for general popularity, allowing factors to learn preference differences. It uses seed 151, learning rate 0.045, regularization 0.002 and ten epochs. The final blend is 80% V11 plus 20% new model.

Settings were selected on January 26 using history-reweighted AUC, then frozen for January 25. Reweighting uses only evaluation history lengths, never evaluation labels.

| First-event validation | V11 | New blend | Gain |
|---|---:|---:|---:|
| January 26 | 0.810155847 | 0.813889863 | +0.003734015 |
| January 25 | 0.797285247 | 0.804728376 | +0.007443129 |

History-reweighted scores improve from 0.802234258 to 0.805837112 and from 0.790605569 to 0.797856273. The independent tester verified exact recipe, event and label alignment, finite probabilities and checkpoint replay. Paired event intervals for the ordinary gains are approximately [0.003234,0.004234] and [0.006809,0.008077]. These conditional intervals do not account for repeated selection on the same historical windows.

## Other experiments

- 32-factor auxiliary weight 0.15 without popularity correction: overall gain failed January 26. A predeclared history-at-most-four gate had tiny gains with intervals crossing zero on both dates; not promoted.
- Nonnegative topics, rank 16: sparse factor fit succeeded, but the subsequent classifier was stopped at 200 trees with validation AUC 0.624659. This was a partial experiment, not a completed model comparison.
- Corrected rank 8: stopped after six epochs, ordinary AUC 0.810477 and reweighted 0.802577; below the accepted rank 16. No confirmation fit or submission.
- Corrected rank 32: stopped after five epochs, ordinary AUC 0.811270 and reweighted 0.803378; below accepted rank 16. No confirmation fit or submission.

## Additional diagnostic

A frozen check on 10,000 deterministically selected non-first January 26 events compared a pre-January-26 V6 validation model with the new factor model. Its frozen 20% blend scored 0.825863 versus V6 0.816311. These are different events and a different baseline from the official first-event task. This diagnostic supports transfer but must not replace the main validation scores or be presented as a leaderboard score.

## Reproduction

First reproduce V11 and prepared histories/timestamps using the existing pipeline. The corrected weighted generator writes audited V11 validation aliases with event keys.

```powershell
python imp2_warm_choice.py --weight 1.0 --dim 16 --pop-correct --epochs 10
python tst2_candidate.py imp2_warm1.0_d16_pc1 --weight 0.2
python imp2_finalize.py --dim 16
python check_submission.py artifacts/imp2_submission_warm1.0_d16_pc1.csv
```

The full-data refit uses all seven training days, retaining the frozen configuration. No automatic competition upload is performed. The user-reported official V11 baseline is 0.801; the new model's official score is unknown. **0.85 has not been reached.**
