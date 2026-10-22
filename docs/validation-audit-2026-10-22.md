# Validation audit: 2026-10-22

The user confirmed **submission_v11.csv received official AUC 0.801**. No other public scores quoted in older local notes have been verified in this run. The target 0.85 has not been reached.

Every original evaluation row and event history was independently compared with prepared numeric arrays: 529,390 rows and 52,939 events passed. The V11 CSV was reconstructed from its component predictions, with maximum decimal-rounding difference 5e-11 and unchanged SHA-256 0bf06c9673d3fa4b36aac4f5f2e3de8bb80c33540883e0aeea9dcf9ea019873e.

Old predictions_v11_day*.npz aliases contained a rejected V10-level experiment. They were archived and replaced with the actual weighted V11 predictions and matching event keys. This did not change a submission CSV. The repository generator now publishes event keys and the correct V11 aliases only after its two-date acceptance gate.

## Distribution diagnosis

Evaluation has histories of at most four clicks in 13.46% of events, compared with 7.27% of January 26 validation events. Full twenty-click histories make up 39.21% of evaluation versus 51.12% of validation. V11 is much weaker with short histories.

Reweighting five validation history-length groups to the unlabeled evaluation proportions changes V11 January 26 AUC from 0.810156 to 0.802234, close to the user-reported official 0.801. January 25 changes from 0.797285 to 0.790606. This supports a possible covariate-shift explanation; it does not prove causation or predict an official score exactly. Evaluation labels are never available or used.

Subsequent experiments report ordinary and history-reweighted AUC. Both dates have been reused for selection. Paired event intervals are conditional on chosen settings and do not include repeated-selection uncertainty.

## Auxiliary-data audit

Earliest-per-user histories precede both their source event and every tested training cutoff. Five repeated exact (user,item,timestamp) records were removed in the new auxiliary objective; repeated items at different times were retained. Source training events are sorted, labels are binary, and every event has at most one positive.

Metrics and row/recipe checks are in results/tst2_*.json. Audit helpers require prepared arrays and prior model predictions in the configured artifact directory. They do not retrain models or upload submissions.
