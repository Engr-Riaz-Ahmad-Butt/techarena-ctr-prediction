# Further exposure-policy tests — 2026-10-29

**Neither fixed recipe passes the arrival-day screening rule. No model is promoted.**

## What changed

The current reported 0.818023 local candidate uses a spectral exposure-policy signal. Its tree/richer-feature follow-ups were already rejected by the parallel session. This round tested two different ways of learning the exposure policy itself, using only actual shown-item counts/triples from before the cutoff. No click outcomes entered either fit. Existing nx/nxa/tst work was read only.

1. A 32-factor hierarchical Gamma-Poisson model with 80 preset variational iterations, shape priors 0.3 and hierarchical means 1. The feature is personal exposure lift, with population and activity mass removed. Source: [Gopalan, Hofman & Blei (2015)](https://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf). The paper supports the count-model approach; it provides no evidence of gains on this competition.

2. A 32-factor signed model that jointly learns user/item logging factors for three preset passes. It distinguishes observed shown items from independent uniform-catalog noise, with a fixed marginal exposure offset and learned user normalizer. The scored feature is the personal factor dot product. Parameters: seed 8231, Adagrad learning rate 0.03/initial accumulator 0.1, anchor 0.01 to cutoff-matched spectral vectors. Source principle: [Gutmann & Hyvärinen (2010)](https://proceedings.mlr.press/v9/gutmann10a.html). This applies contrastive estimation to exposure density. The earlier closed NCE experiments predicted clicks; the previous cheap policy proxy adjusted users in a fixed item space. Here the user and item logging factors both learn. Noise examples are statistical density-class examples, not claims of unclicked ads or missing warm-up exposures.

Both implementations were written here from the mathematical ideas; no repository code or external data was used. No additional calibration head, blend-strength search, seed search, or short-history specialist was fitted.

## Matched validation

The frozen 9,878-user sample contains one first-post-cutoff event per user and ten aligned candidates. Arrival counts are 6,919 / 1,794 / 856 / 309. Models freeze after the first three training days; the same model scores all four arrival days. No holdout events are in this cohort. Joint IPF event weights match arrival 67:17:7:4 normalized over four supported days and history 1.26% / 16.37% / 82.37%. Evaluation arrival days 5–7 remain unsupported.

The original reference is the frozen simplified V53-component horizon blend, AUC **0.759031195**. It is not a full V53 rebuild or nxa_h1 replay. The SVD exposure control uses the fixed typ16/typ32/cos32 recipe with IDF-log normalized counts, 64 spectral components and seed 4403. Historical CDFs come from 7,047 first-user day-2 events, scored by prefix-2 models. The current validation file does not supply percentiles or fitted normalization.

Control = original reference + 0.015 × standardized-within-event SVD exposure signal. Candidate = the same original reference + 0.015 × the corresponding new exposure signal. The score bridge, cutoff, cohort, weights and historical normalization block are fixed. This tests replacing the exposure model, without fitting a new head. The same SVD control scores are used in both tests.

Screening requires +0.0002 overall **and each of four arrival days**, against both the SVD control and original reference. Full-cohort +0.0005 confirmation is allowed only for survivors. Both recipes failed screening, so confirmation was not run. Intervals are 200-draw paired user bootstraps with fixed models and weights; they omit recipe-selection, factor-refitting and repeated-validation uncertainty.

## Gamma-Poisson count model — reject

| Segment | Events | SVD control AUC | New model AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759536768 | 0.759292110 | -0.000244658 | [-0.000677232, +0.000146089] |
| Arrival 1 | 6,919 | 0.767560086 | 0.767273577 | -0.000286509 | [-0.000784482, +0.000168420] |
| Arrival 2 | 1,794 | 0.737056816 | 0.737050183 | -0.000006632 | [-0.001087907, +0.000987383] |
| Arrival 3 | 856 | 0.738949363 | 0.738970657 | +0.000021294 | [-0.001447513, +0.001273339] |
| Arrival 4 | 309 | 0.756799820 | 0.756199894 | -0.000599927 | [-0.003862232, +0.001752104] |
| Arrival 3+ | 1,165 | 0.744870236 | 0.744630202 | -0.000240034 | [-0.001540363, +0.000935609] |
| History 0 | 160 | 0.627021345 | 0.626961188 | -0.000060157 | [-0.012313740, +0.010822981] |
| History 1–5 | 1,825 | 0.675743692 | 0.675410355 | -0.000333338 | [-0.001330715, +0.000650309] |
| History 6–20 | 7,893 | 0.774128125 | 0.773911132 | -0.000216993 | [-0.000676475, +0.000337677] |

Runtime 154.7 seconds locally; final fixed checkpoint only.

## Learned contrastive policy — reject under arrival-day gate

| Segment | Events | SVD control AUC | New model AUC | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| Overall | 9,878 | 0.759536768 | 0.759759776 | +0.000223008 | [+0.000025950, +0.000410513] |
| Arrival 1 | 6,919 | 0.767560086 | 0.767877674 | +0.000317588 | [+0.000103914, +0.000537600] |
| Arrival 2 | 1,794 | 0.737056816 | 0.737059501 | +0.000002686 | [-0.000490511, +0.000509735] |
| Arrival 3 | 856 | 0.738949363 | 0.739028558 | +0.000079195 | [-0.000412774, +0.000592504] |
| Arrival 4 | 309 | 0.756799820 | 0.756727964 | -0.000071857 | [-0.001162712, +0.000777724] |
| Arrival 3+ | 1,165 | 0.744870236 | 0.744875220 | +0.000004984 | [-0.000484588, +0.000529909] |
| History 0 | 160 | 0.627021345 | 0.628823120 | +0.001801775 | [-0.001921268, +0.005833661] |
| History 1–5 | 1,825 | 0.675743692 | 0.675532262 | -0.000211430 | [-0.000659348, +0.000224024] |
| History 6–20 | 7,893 | 0.774128125 | 0.774442507 | +0.000314382 | [+0.000071081, +0.000537483] |

Runtime 38.7 seconds locally; final fixed checkpoint only.

## Interpretation

The Gamma-Poisson model gains +0.000261 over the original reference but loses -0.000245 against the stronger SVD exposure control. It supplies no accepted improvement.

The learned policy reaches **0.759759776** versus **0.759536768** for SVD, a conditional **+0.000223008** overall (paired 95% [+0.000025950, +0.000410513]). It gains +0.000728581 over the original reference, which includes the existing exposure signal’s contribution. This is a small positive observation, not an accepted model: arrival 1 gains +0.000317588, arrival 2 +0.000002686, arrival 3 +0.000079195, and arrival 4 -0.000071857. Arrivals 3+ combined gain only +0.000004984. History 1–5 also loses -0.000211430 versus SVD. No gate is relaxed and the failed recipe is not tuned or expanded.

Later-arrival estimates are noisy, especially the 309-event fourth day. This does not prove that the method is intrinsically worse later or establish a general performance ceiling. It does show that this test has not demonstrated the required gains across the agreed horizon.

The earlier-cutoff horizon AUC cannot be added to or compared numerically with the reported 0.818023 local result. The full V53/nxa_h1 chain was not reproduced here. The last confirmed official score in the ledger is still 0.816623; no official improvement or target attainment is claimed.

## Audit and artifacts

Saved models reproduce the new signal arrays on 128 events. Changing candidate labels leaves all signals identical; permuting candidate order permutes scores correspondingly. All predictions are finite/aligned; both tests share identical controls. Fitting uses user/item/time only, with maximum timestamps strictly earlier than prefix-2/prefix-3 cutoffs. Variational count allocations conserve observed counts, its ELBO rises at every logged checkpoint, and contrastive updates pass finite-difference checks including a noise/data item collision. Input/source manifests and all 29 protected artifacts are unchanged. No holdout or evaluation data was scored, no submission was created, and no upload was attempted.

Full numerical reports, frozen plans, predictions and checkpoints are in `artifacts/v54_horizon/cdx_policy_hpf_001/` and `cdx_policy_contrast_001/`. Post-fit replay is recorded in the latter’s `postfit_audit.json`.
