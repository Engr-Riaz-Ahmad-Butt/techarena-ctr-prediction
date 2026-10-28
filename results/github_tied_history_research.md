# Same-challenge public-code research, 2026-10-28

Previous goal turn classification: progress. It implemented and tested the
deduplicated causal click ledger; this follow-up added the tied-vector model and
a bounded real-data pilot. The 0.83 goal remains unachieved.

Primary source: https://github.com/jancastachova/techarena2026-atu
Inspected revision: 1ed9fd299e4a04352279272bd6be02ed1d08ceec.
Task matched by Phase 1 title, exact history columns/example rows, and the
Ireland submission portal. This establishes a same-task solution, not a copy of
our implementation. Repository/name searches and global code searches found no
independently verified same-task public 0.83+ solution.

Evidence:
- `experiments/RESULTS.md`: author-recorded public Experiment 11 AUC 0.815842.
- `experiments/resultsSIMPLE.md`: newer approximate Experiment 13 public AUC
  0.818; separate local baseline 0.83571. No independent leaderboard verification.
- `experiments/15_more_seeds/more_seeds.py`: combines 15 additional seeds with
  Experiment 13, 30 total, using full click histories, tied item vectors,
  click-log auxiliary negatives sampled by exposure, and LightGBM.
- Our directly confirmed official V35 score remains 0.816623. Ordinary local,
  adjusted local, and reported public scores are not interchangeable.

Local source comparison:
- `imp18_nce_core.py`: separate U/V/H tables; real-history context is sqrt-scaled.
- `imp7_sequence_core.py`: earlier-only auxiliary sequence contexts and distinct
  history/candidate vectors, plus user vectors.
- `core_v6.py`: aggregate positive profiles already use more than the current
  history window. Consequently, full-history information alone is not new.
- `imp15_als.py`: exposure-aware user/item ALS, not a tied item-history model.

The distinct hypothesis is the combination of one shared item table, mean
full-history profile without a learned per-user vector, and prefix-only
leave-one-click-out auxiliary training. Do not describe the bounded pilot as
testing that entire combination: it currently has impression BCE only.

Implementation checks passed: click deduplication, temporal boundaries,
leave-one-click-out exclusion, numerical gradients with duplicate IDs shared
between candidate and history roles, empty histories, and balanced-popularity
synthetic preference learning.

Pilot: `kaggle/tied_history_pilot.py`, seed 7103, 40,000 uniformly sampled fitting
events before day6, 8 epochs, 32 factors, 2 CPU threads. Ledger uses the entire
fitting prefix; held-out labels never enter fitting. Validation is the unchanged
V53 day6 cohort, already reused, exploratory only. Checkpoint replay covers the
first 100 events, not every prediction. Artifacts are stored separately under
`artifacts/tied_history_pilot_001/`; no submission is produced.

Before a full candidate: add the auxiliary exposure-negative objective, persist
the fixed recipe and source/input hashes, produce temporal out-of-fold factor
features for any tree stack, and compare the same fixed recipe on both dates.
Fresh labeled holdout remains unavailable and required by the current promotion
plan. A pilot's low standalone AUC cannot reject the entire unimplemented recipe;
its high score cannot establish generalization or an official score either.

## Observed pilots

Pilot 001: ordinary AUC 0.5049471964 versus V53 0.8254233319 on identical
day6 rows. Global Adam weight decay 0.0001 collapsed item-vector RMS to
7.28164e-10. Rejected. Runtime 86.19 seconds.

Pilot 002: same data sample, seed, epochs and learning rate; global weight
decay removed as a controlled learning diagnostic. Ordinary AUC 0.5545576073,
item-vector RMS 0.6240823. Training BCE dropped from 0.2379753 to 0.0367322,
but generalization remains very weak, consistent with memorization of the small
sample. Rejected. Runtime 86.86 seconds. Neither pilot tests auxiliary click-log
training or the complete out-of-fold LightGBM pipeline; neither is a competitive
candidate. No threshold crossing or official improvement is claimed.

Next development step: implement exposure-sampled leave-one-click-out auxiliary
training and normalize regularization over the actually sampled vectors, then
use the full fitting prefix rather than enlarge the small-pilot tuning search.
