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

## Full-prefix auxiliary experiment 001

Completed 2,000 fixed steps, seed 7201, with 588,873 fitting events and 945,342
prefix-only auxiliary clicks. Auxiliary negatives follow fitting-item exposure
counts. Item/candidate/history vectors share parameters. Sampled-vector L2
replaces the failed global Adam weight decay. No validation early stopping or
blend search. A frozen 5% rank blend produced:

| Metric on the same reused day6 rows | V53 | Fixed blend |
| --- | ---: | ---: |
| Ordinary ROC AUC | 0.8254233318844953 | 0.8256605136652951 |
| History-adjusted ROC AUC | 0.8171690462350286 | 0.8173907383647875 |

Adjusted gain +0.0002216921 is below the required +0.0005, so no day5 fit,
promotion, or submission. Standalone factor AUC is 0.7661169667 ordinary and
0.7568110852 adjusted; its small complementary signal does not establish that
the complete temporal out-of-fold tree pipeline has been tested.

Independent audit passed input/source hashes, all-row NumPy equation replay
(maximum probability error 2.206e-7), exact full blend/metric replay, and 24
raw-array fitting-history context oracles. Auxiliary causality was not separately
rebuilt from raw arrays by the independent audit; prefix construction and unit
tests provide narrower supporting evidence. Full Torch checkpoint replay was
exact for every validation row.

Recorded wall time 2,677.95 seconds includes a discontinuity: elapsed time jumped
from 209.06 seconds at step400 to 2,256.98 at step500. Cause is unverified; this
must not be presented as measured CPU time or GPU usage. Recent 100-step blocks
took roughly 20–30 seconds. Two CPU threads were used, no GPU was requested.

Reports: `results/tied_history_full_001.json`, `_audit.json`, `_frozen.json`.
The original executed runner is preserved as
`results/source_snapshots/tied_history_full_001.py`. The current runner's day5
cohort assertion was subsequently corrected to match V53's existing daily
horizon; day6 behavior is unchanged. No day5 result was available at that stage.

The 0.83 objective remains unachieved. A further development candidate would
need temporal out-of-fold factor features for a tree learner, not tree fitting
on in-sample factor predictions. The fresh-holdout promotion gate still applies.

## Day5 factors and matched tree stack

The previous goal turn made concrete progress: full-prefix auxiliary fitting and
its independent audit completed. This turn completed the same frozen factor
recipe on day5, followed by a fixed matched-control tree experiment. No training
remains active after completion.

Day5 used 490,420 fitting events and 880,465 auxiliary clicks before its cutoff.
Fixed 5% rank blend: ordinary AUC 0.8185102941, adjusted AUC 0.8113809781,
adjusted gain +0.0004695928 versus V53. Recorded wall time 1,248.83 seconds.
All-row checkpoint replay and independent NumPy replay passed (maximum
probability error 2.016e-7), as did 24 raw-history context oracles and hashes.
Together with day6, this fixed blend improves both observed cohorts, but fails
the +0.0005-per-cohort gate and remains exploratory.

Tree settings were frozen before fitting in `results/tied_history_stack_plan.json`:
300 trees, depth4, learning rate0.03, L2=50, two user folds, seed7301, 10% rank
blend. Control uses seven fixed-parent/history/time features; treatment adds 13
factor features. Factor features are invariant to orthogonal rotations of each
date's independently trained item table; unit checks cover this, candidate
permutations and empty histories. Actual day6 feature alignment also passed.

| Reused cohort / adjusted AUC | V53 | Matched control | Factor treatment |
| --- | ---: | ---: | ---: |
| Day5 conditional meta-user-crossfit | 0.8109113853 | 0.8109285750 | 0.8111045913 |
| Day6 future scoring | 0.8171690462 | 0.8172286522 | 0.8174056264 |

Treatment day6 ordinary AUC is 0.8256658662. Adjusted gain over control is
approximately +0.000177 on each cohort, but the gain over V53 is below +0.0005.
The tree treatment adds only +0.000014888 adjusted AUC over the simpler fixed
factor blend on day6 and is worse than that blend on day5; it is not a decisive
replacement. V53 remains the accepted baseline. No CSV was exported.

All six saved CatBoost models replayed exactly on all their scored rows; an
independent saved-output audit recomputed every reported metric and checked
input hashes, the frozen plan/source hash, and baseline event/label identity.
Runtime for both tree arms, feature preparation and scoring was 71.86 seconds.
The day5 meta crossfit conditions on previously frozen parent OOF arrays; parent
ancestors were not retrained inside each meta fold. This is not fully nested CV
or another untouched temporal validation fold. Both dates were already reused.

## Cloud reuse evidence

`results/kaggle_tied_history_input_compatibility.json` verifies the existing
81.9MB neural bundle against all 686,824 original events: users, ten candidates,
histories, bucket, day type, timestamp, target labels, and both validation cohorts'
indices/labels/baseline scores match exactly. Every valid original integer
history timestamp is recovered exactly by rounding timestamp minus stored
float32 age in days times86400. This was checked against the actual raw arrays;
maximum recovered timestamp error is zero seconds. It does not add new labeled
dates. A GPU implementation can reuse the already uploaded bundle rather than
upload another raw CSV. Browser inspection confirmed the notebook is available;
no new remote runtime was started and no new GPU hours are claimed.

Next high-value development direction is a predeclared multi-seed factor average
on Kaggle GPU, comparing against its matched single-seed GPU control. Do not
launch several long local CPU seed jobs or select lucky seeds on day6. These
development ideas do not remove the fresh-holdout promotion requirement or prove
the 0.83 target.

## Fixed five-seed GPU experiment

Completed on a verified Tesla T4 with PyTorch2.11.0+cu128. Seeds7201–7205,
2,000 steps each, the same 32-factor objective and a frozen 5% rank blend.
The efficient embedding lookup/fused regularizer matched the earlier duplicate-
safe loss and gradients in double-precision CPU tests. Prefix/future-label tests
passed. All seven prepared day6 arrays matched the audited CPU arrays exactly;
the same raw-array SHA256 constants were checked in Kaggle before fitting.

| Reused day6 metric | V53 | Matched single GPU seed | Five-seed mean |
| --- | ---: | ---: | ---: |
| Ordinary AUC | 0.8254233319 | 0.8256605106 | 0.8257031575 |
| History-adjusted AUC | 0.8171690462 | 0.8173907355 | 0.8174582056 |
| Standalone factor ordinary AUC | — | 0.7661169669 | 0.7825589859 |

No seed was selected on validation scores. Averaging improved the raw factor,
but adjusted gain over V53 was only +0.0002891594, and over the matched seed
only +0.0000674701. The predeclared +0.0005 primary gate failed, so no day5
five-seed run, promotion or CSV export occurred. The 0.83 goal remains unmet.

All five checkpoint predictions replayed exactly on GPU. Local independent
NumPy replay of all five recovered models passed on every validation row with
maximum probability errors below 2.27e-7. All five probability arrays, their
fixed mean, the bag blend, baseline row/label identity, history-distribution
weights and reported AUCs were independently checked. One numerical limitation
is explicit: the stored single-control blend differs from this environment's
float64 reconstruction by at most 3.73e-9; its cause was not independently
established. It is not labeled bit-exact. The bag blend is exact, and all saved
score metrics match the report within1e-12. This small discrepancy does not
change the failed primary gate.

### Output preservation and recovery

The first run took125.64 seconds. Browser file-download attempts failed, and an
interactive-session reset subsequently removed the working directory (zero
models remained). Its rendered summary was preserved as observed evidence.
Recovery reran the same frozen recipe in121.37 seconds and produced identical
reported metrics. This was recovery from proven file loss, not a lucky-seed or
score-selection retry. The first8MB incomplete archive was renamed `.partial`.

The recovered audit ZIP (17,545,524 bytes) and model-only ZIP (30,719,036 bytes)
were exported through rendered notebook text in bounded parts. Per-part and
whole-file SHA256 checks passed, as did all five checkpoint file hashes against
the cloud evidence. Models and arrays are now under
`artifacts/tied_history_gpu_bag_002/`, excluded from Git. Full provenance is in
`results/tied_history_gpu_bag_002_preservation.json`. Normal downloads should not
be retried in long timeout loops; preserve compact evidence first.

Kaggle Quick Save version9, "Fixed five-seed GPU audit", completed with visible
cell outputs. Durable model-file storage in that saved version is not claimed;
the local verified bundles provide preservation. The idle GPU session was then
explicitly stopped and "Session stopped" observed. Total allocated GPU session
hours were not independently tallied. A proof image is saved beside the project
as `kaggle-tied-history-bag-result.png`.

The standalone three-cell notebook now builds compact audit and model bundles
and avoids the broken absolute `FileLink` download path. Use Save & Run All only
on that standalone notebook, not the older draft containing rejected trials.

Further seed averaging has limited complementary value at this point. A larger
untested step is to construct per-day factor features strictly before each
training day and train a pointwise tree on all prior-date rows, rather than only
the day5 first-event meta cohort. This must preserve temporal OOF boundaries,
predeclared settings, matched controls and the fresh-holdout promotion rule.
