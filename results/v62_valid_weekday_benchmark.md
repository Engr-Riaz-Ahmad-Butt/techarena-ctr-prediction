# Valid weekday benchmark audit

## Benchmark
- Metric: ROC AUC
- Accepted repo protocol: day5 user-crossfit + day6 future-date validation
- Target: 0.85

## Result
- Status: REJECTED
- Reason: candidate predictions are scored on the same retained labels/cohort as the V53 replay, so they are not a valid future benchmark.
- Target achieved: no

## Comparison

| Day | Baseline AUC | Candidate AUC | Gain | Valid under protocol |
|---|---:|---:|---:|---|
| 5 | 0.818091 | 0.833840 | +0.015749 | False |
| 6 | 0.825423 | 0.842950 | +0.017527 | False |
