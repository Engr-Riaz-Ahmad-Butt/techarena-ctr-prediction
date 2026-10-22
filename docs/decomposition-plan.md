# V17 improvement experiment: warm-factor decomposition

Target requested by user: at least 0.82 AUC. No local or official score is promised.

Baseline: submission_v17.csv; local January 26 AUC 0.816134690306832, January 25 AUC 0.8073751845446917. Official V17 AUC is unknown; the latest user-confirmed official result remains V11 at 0.801.

Predeclared configuration:

- Add 32 scalar factor-decomposition features to the existing 194-feature stacked classifier.
- Use the same pre-cutoff warm16 checkpoint for each block, preserving chronological feature isolation. Validate all newly generated rows against the existing warm raw-score cache.
- Features describe user/item biases, separate stable-user and recent-history dot products, norms/cosines, recent-click similarities, no-click propensity and event-relative differences.
- Fixed classifier settings: 599 trees, depth 7, learning rate 0.08, seed 42. No tree count or classifier hyperparameter tuning in this experiment.
- On January 26 select only how much the new component replaces the original warm-feature component: 0%, 50% or 100%, using history-adjusted AUC. Keep the remaining V17 recipe fixed.
- Freeze the selected mixture for January 25. Require improvement on both ordinary and history-adjusted AUC on both dates, and a positive conditional paired event-level interval on the selection date.
- Two dates have been reused repeatedly; calculated intervals do not account for model-selection bias. Any official score requires competition scoring.
- Preserve V17. Train a final component and create V18 only if the candidate passes independent review.

Source: imp4_decompose.py. Independent audit source: imp4_test_decomposition.py. Reports and predictions use the imp4 prefix.
