# Preference research — 30 October 2026

**The 0.82 target has not been reached. Both new fixed recipes fail screening and are rejected.**

The working target is the same local adjusted metric as the reported 0.818023 candidate, a gap of about 0.001977. The last user-confirmed official score in the project ledger remains 0.816623. This round uses the earlier-cutoff matched horizon to screen new signals. Its approximately 0.759 AUC is a different cohort, weighting and training-window length; gains cannot be added to 0.818023. No full V53/nxa_h1 or official-score improvement is claimed.

## Shared exposure/click item representation

New question: can supervision from the logging policy improve the underlying click representation when it updates shared item factors, rather than acting as a separate expert? The general source principle is parameter sharing across related relations in [Singh & Gordon (2008), Collective Matrix Factorization](https://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf). The particular exposure/click objective here is an original adaptation, not that paper’s implementation or evidence of competition gains.

Both arms use the exact existing warm16 kernels: 16 dimensions, ten epochs, seed 151, learning rate 0.045, regularization 0.002, warm-positive auxiliary weight 1 and popularity correction. Prefix-2 initial statistics precede the prefix-3 scoring cutoff. Fitting uses all 295,114 earlier events and 557,429 genuine earliest-history positive pairs. Each arm also learns a private exposure-user head from the 2,951,140 observed impression triples and independent uniform-catalog statistical noise (weight 0.1, seed 8531). Only the treatment enables its data gradients into the shared click-item V; no additional shared-V regularization is added. No missing warm-up impressions are imputed.

The control **reproduces every archived warm16 parameter bit-for-bit** before validation metrics are calculated. Thus the original warm-history and event-choice architecture, optimizer and calendar are retained. Existing project functions are reused as definitions to avoid module side effects; new auxiliary code was written here and passed independent finite-difference, frozen-V and zero-weight checks.

Runtime: 311.7 seconds (5.2 minutes), below the conservative 11.6-minute pilot estimate.

## Two-interest representation diagnostic

A second new question: does a single history mean conceal two coherent interests? [Li et al. (2019), MIND](https://arxiv.org/abs/1904.08030) motivates multiple user representations. This test is an original **static diagnostic**, not MIND training, capsule routing, learned attention or a sequence model. Existing warm16 factors remain frozen.

Split the at-most-20 history H vectors into two deterministic Euclidean clusters: sorted item IDs, farthest-pair initialization, at most six Lloyd steps. Component weights are history cluster shares. Their weighted mean exactly conserves the original preference profile. Average their conditional ten-item softmax probabilities, keeping the original event-click probability fixed. Histories shorter than two or with degenerate identical vectors are unchanged. No parameters, temperature, gates or strengths are fitted. History order is irrelevant.

It splits 47,223 of 49,420 permitted horizon events. Runtime 44.6 seconds; the 16.5-second pilot estimate omitted an extra compiler specialization used by the permutation check.

## Frozen validation and decision

Same frozen 9,878-user sample, one first-post-cutoff event per user, ten candidates. Arrival counts: 6,919 / 1,794 / 856 / 309. IPF weights match arrival 67:17:7:4 normalized over four supported days and history 1.26% / 16.37% / 82.37%. Evaluation days 5–7 remain unsupported. Weights and cohorts are unchanged. Supplied history clicks after the cutoff remain usable only when strictly earlier than their event.

The unchanged simplified bridge is 95% warm-expert rank plus 5% frozen-ALS rank. Ranks are calculated jointly over the full permitted horizon without inspecting its labels; only the 20% sample’s outcomes are assessed. This is not a full V53/nxa_h1 replay. The tiny control AUC difference from the older 0.759031195 reflects the original compiled scorer versus the prior NumPy scorer; both new tests share the same control, 0.759031201.

Require +0.0002 overall and on each of four arrival days, followed only for survivors by +0.0005 full-cohort confirmation. Neither survives; no full-cohort outcome check or parameter search was run. Confidence intervals use 200 paired user-bootstrap draws, conditional on fixed models and weights, and omit repeated-validation and model-selection uncertainty.

### Shared representation — REJECT

| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759031201 | 0.758751566 | -0.000279634 | [-0.000410215, -0.000149736] |
| Arrival 1 | 6,919 | 0.767066938 | 0.766765776 | -0.000301163 | [-0.000432263, -0.000128888] |
| Arrival 2 | 1,794 | 0.736302723 | 0.736045923 | -0.000256799 | [-0.000589194, +0.000017406] |
| Arrival 3 | 856 | 0.738793692 | 0.738497401 | -0.000296291 | [-0.000730525, +0.000163745] |
| Arrival 4 | 309 | 0.756915154 | 0.756771199 | -0.000143956 | [-0.000934282, +0.000571126] |
| Arrival 3+ | 1,165 | 0.744796930 | 0.744537697 | -0.000259233 | [-0.000602261, +0.000173123] |
| History 0 | 160 | 0.628417571 | 0.628118093 | -0.000299477 | [-0.000949747, +0.000438677] |
| History 1–5 | 1,825 | 0.675029475 | 0.674874825 | -0.000154650 | [-0.000446195, +0.000127872] |
| History 6–20 | 7,893 | 0.773689289 | 0.773385767 | -0.000303522 | [-0.000453639, -0.000173079] |

### Two-interest mixture — REJECT

| Segment | Events | Control AUC | Candidate AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759031201 | 0.757435768 | -0.001595433 | [-0.002749275, -0.000406804] |
| Arrival 1 | 6,919 | 0.767066938 | 0.765527339 | -0.001539599 | [-0.002737474, -0.000047908] |
| Arrival 2 | 1,794 | 0.736302723 | 0.733521661 | -0.002781061 | [-0.005103471, -0.000084214] |
| Arrival 3 | 856 | 0.738793692 | 0.736810933 | -0.001982759 | [-0.005595815, +0.001508204] |
| Arrival 4 | 309 | 0.756915154 | 0.759311770 | +0.002396616 | [-0.005071063, +0.011378778] |
| Arrival 3+ | 1,165 | 0.744796930 | 0.744252180 | -0.000544749 | [-0.004560107, +0.003112142] |
| History 0 | 160 | 0.628417571 | 0.628312481 | -0.000105090 | [-0.000376451, +0.000181366] |
| History 1–5 | 1,825 | 0.675029475 | 0.676124539 | +0.001095064 | [-0.001405495, +0.004115256] |
| History 6–20 | 7,893 | 0.773689289 | 0.771597636 | -0.002091653 | [-0.003335784, -0.000936672] |

## Findings and audit

Sharing exposure gradients reduces AUC on every arrival day, overall -0.000279634 (paired 95% [-0.000410215, -0.000149736]). Lower auxiliary training loss did not improve click generalization. This rejects the fixed shared-representation recipe, without proving that all forms of multitask recommendation are ineffective.

The two-interest mixture loses -0.001595433 overall (paired 95% [-0.002749275, -0.000406804]). History 1–5 and arrival 4 have favorable point estimates, but their intervals include losses; they do not justify a subgroup specialist or a change to the gate. This rejects the static diagnostic and is not a full MIND ablation.

Post-fit audit passes: exact control parameter reproduction; saved probabilities replayed on 128 events; changing labels leaves predictions unchanged; strict history timestamps; one event per user; identical controls; mixture event mass and degenerate cases preserved; history/candidate order checks; finite predictions and correct weight marginals; all 29 protected files and source/input manifests unchanged. No holdout or evaluation was scored and no submission was created.

Reports, frozen plans, models and predictions: `artifacts/v54_horizon/cdx_collective_001/` and `cdx_multi_interest_001/`. The replay audit is `cdx_collective_001/postfit_audit.json`. Results are logged. The current candidate is retained; 0.82 remains unverified. The actual competition/Instructions URL was requested to check official details before choosing another new signal. No target attainment or general statistical ceiling is claimed.
