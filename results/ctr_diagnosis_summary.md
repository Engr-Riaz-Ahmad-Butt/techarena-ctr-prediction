# CTR error diagnosis and targeted experiment

The official result remains V18 at 0.809189; the user confirmed no later upload. V23 is the strongest accepted local model, not an officially scored submission.

## Evidence

- Jan26 AUC is 0.746552 for 1–4 historical clicks and 0.841814 for 20 historical clicks. Subgroup AUC is not overall AUC and is not an additive explanation of its gap.
- Users with 1–4 clicks represent 6.71% of Jan26 validation and 12.21% of evaluation. Users with a last click more than seven days ago represent 3.45% and 10.35%, respectively.
- Short/old histories are therefore a plausible priority, not proof of the cause of all errors. Evaluation also has fewer low-exposure items, which may offset some difficulty. Labels for evaluation are unavailable.
- Saved rank-blend outputs are ranking scores, so calibration metrics would be misleading without calibration. Diagnosis uses AUC and event ranking metrics.

## V31 experiment

Reweight all-event training using label-free evaluation history-length/recency distributions, with smoothed and clipped density ratios. Keep architecture, one training epoch, and 10% blend fixed. Checkpoint and input hashes and prediction replay were verified; group boundary handling was tested.

| Local date | V23 | V31 |
| --- | --- | --- |
| Jan26 | 0.824657367 | 0.824712888 |
| Jan25 | 0.817399619 | 0.817439098 |

Both ordinary and history-weighted point estimates improve. Jan25 conditional intervals still include zero, so V31 remains an exploratory candidate and does not replace V23. Validation dates have been repeatedly reused; intervals exclude model-selection uncertainty. No new evaluation CSV was generated and 0.84 has not been achieved.

The next useful research direction is stronger personalization for sparse/old histories, evaluated with distribution-aware metrics and a new temporal validation run. More tiny blend sweeps on these same dates would give weak evidence.
