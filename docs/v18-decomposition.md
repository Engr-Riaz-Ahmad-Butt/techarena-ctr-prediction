# V18: decomposed warm-factor preferences

The preceding V17 classifier received a warm-factor click probability, logit, relative score and rank. It could not distinguish user preference, recent-history preference, item bias and no-click propensity when those terms produced the same scalar score. This experiment adds 32 causal decomposition features and refits the classifier with the unchanged 599 trees, depth 7, learning rate 0.08 and seed 42.

Every training block uses a warm-factor checkpoint fitted strictly before its day. Reconstructing its softmax probabilities from the decomposition agrees with the existing feature cache within floating-point tolerance. The selected component replaces the original warm-feature classifier; V6, V7, event competition adjustment and the three warm-factor seed models retain their V17 weights.

| Validation | V17 AUC | Candidate AUC | V17 history-weighted AUC | Candidate history-weighted AUC |
|---|---:|---:|---:|---:|
| January 26 | 0.816134690 | 0.816800294 | 0.807982616 | 0.808540088 |
| January 25 | 0.807375185 | 0.808296773 | 0.800304001 | 0.801149062 |

January 26 selected complete component replacement from the predeclared 0%, 50%, 100% grid using history-weighted AUC. That choice and all classifier settings were frozen for January 25. Independent replay produced exactly the saved predictions on both dates. All four paired event-level conditional improvement intervals are positive. These dates have been reused for model development; the intervals exclude repeated-selection bias and evaluation distribution uncertainty.

The requested 0.82 threshold has not been reached locally. The official competition score is unknown. The latest directly confirmed official score remains V11 at 0.801.

After reproducing V17 and its causal caches:

```powershell
python imp4_decompose.py
python imp4_test_decomposition.py
python imp4_finalize.py
```

The finalizer refuses to run unless both validation dates improve and the independent audit agrees. It trains on blocks 4, 5 and 6, uses the cut7 checkpoint for evaluation, and verifies that its final component inputs reconstruct V17 before writing `submission_v18_candidate.csv`. A separate final-file audit is required before copying that candidate to `submission_v18.csv`. Preserve V17; generated artifacts are excluded from Git.

Plan: [decomposition-plan.md](decomposition-plan.md). Metric reports are in `results/imp4_decompose_report.json` and `results/imp4_test_decomposition_report.json`.
