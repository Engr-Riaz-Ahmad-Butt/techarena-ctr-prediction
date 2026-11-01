# Matched control: warm-user anchored ALS refresh

Decision: stop Idea1. No new post-cutoff clicks exist in the frozen first-event cohort, and every warm-user history click was already represented in the cutoff fitting ledger. Candidate and control scores are identical on every row. No full V53 rebuild is scheduled from this result.

## Matched setup

49,420 first-post-cutoff users; 44,627 warm users. 9,878-user screening sample. Cutoff after three training days. Item vectors, user/item biases and global offset fixed. Anchor16, history half-life2days frozen before outcomes. New evidence would include only unique post-cutoff positive clicks strictly before the event; zero such clicks were found. The independent ledger also verified zero missing pre-cutoff history clicks among604,857 warm-user history records. No fake negatives were generated.

ALS control uses the existing causal cutoff3 ALS16 checkpoint. Cold users keep their cutoff vectors identically in both arms; this intentionally avoids invoking the old cold fold-in approximation that introduces uniform background negatives. Thus this is a controlled warm-refresh experiment, not a replay of full V53.

The simplified blend is fixed95% rank(warm16 history-preference expert) +5% rank(ALS). Warm16 is an existing V53 contributor, includes supplied event history, and stays identical in both arms. Ranks computed once across the full four-day horizon, not within arrival day. No new stacker or propensity model was fitted.

## Full-cohort weighted AUC

Weights jointly match four-day arrival proportions67:17:7:4 normalized over supported days, and history proportions1.26%/16.37%/82.37%. Subgroup AUC uses the corresponding frozen weights, retaining the same reference distribution. Evaluation days5–7 remain unsupported.

| Group | Users | ALS control = refresh | Simplified blend control = refresh | Paired delta, 95% interval |
|---|---:|---:|---:|---:|
| overall | 49,420 | 0.724785 | 0.755280 | 0.000000 [0,0] |
| arrival1 | 34,912 | 0.736281 | 0.765746 | 0.000000 [0,0] |
| arrival2 | 8,988 | 0.703239 | 0.735905 | 0.000000 [0,0] |
| arrival3plus | 5,520 | 0.685345 | 0.716654 | 0.000000 [0,0] |
| history0 | 693 | 0.576709 | 0.530806 | 0.000000 [0,0] |
| history1to5 | 9,265 | 0.641382 | 0.675776 | 0.000000 [0,0] |
| history6to20 | 39,462 | 0.739261 | 0.769611 | 0.000000 [0,0] |

20% screen overall: standalone_ALS control=0.728138, refresh=0.728138; simplified_blend control=0.759031, refresh=0.759031.

Every paired user-bootstrap difference is exactly zero because the two per-row score arrays are identical. Actual400-resample user bootstraps on the screening overall and late-arrival cohorts confirm [0,0]. This is a degenerate paired-difference interval for this experiment, not a claim that baseline AUC itself has no uncertainty.

## Meaning and limits

Longer waiting after the cutoff does not automatically supply more observed clicks: each selected event is the first event for that user after the cutoff. Here, the later arrival histories contain no new clicks to update from. Reusing their pre-cutoff clicks as fresh evidence would double-count information. This explains the exact zero result without needing a full ensemble rebuild. A separate proposal to reweight old evidence would be a different experiment, not a successful refresh.

These AUCs are standalone ALS and a deliberately simplified factor-expert blend, not V53 scores. No positive leaderboard gain is claimed. Full V53 has not been rebuilt; Idea2 and Idea3 have not started.

Model/score inputs and29 protected files passed hashes. History timestamps and independent training-click-ledger coverage checked. Kernel tests passed. No holdout opened/scored, no submissions written, no external data, no copied repository code.

Scoring took8.734 seconds. Actual bootstrap/report timing is in report.json. The reporting-only recovery fixed a2e-12 numerical tolerance assertion; it reused already frozen prediction arrays.

Reproduce: `python horizon_matched_control.py` in a fresh output namespace; for the existing completed run, inspect `artifacts/v54_horizon/matched_refresh_001/report.json` and its frozen scores. Do not rerun a completed experiment to select a favorable outcome.
