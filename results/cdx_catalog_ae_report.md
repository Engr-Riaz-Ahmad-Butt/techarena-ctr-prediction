# Catalog autoencoder: confirmed matched-control gain on Kaggle

The fixed denoising autoencoder (DAE) improves the simplified causal horizon control by **+0.001460 weighted AUC** on the full49,420-user cohort. It clears the agreed screening and confirmation gates overall and on every supported arrival day. The variational arm also passes, but adds no advantage over the simpler DAE.

**This is a survivor for subsequent full-pipeline integration. It is not a measured improvement over fullV53/nxa_h1, and does not establish an official0.82 score.** Do not add this delta to0.818023 or any existing leaderboard score. No baseline, V53, submission or official-score ledger was changed.

## Matched results

Both arms were preregistered against the same frozen exposure-enhanced horizon reference. Same initialization,16dimensions,20epochs,50%input corruption,Adam0.001,batch256,seed8711; same training-user batches and corruption RNG. VAE alone adds latent noise and a learned variance/KL term, cap0.2 annealed over5epochs. No validation epoch/seed/weight/temperature selection.

|Cohort|Control|DAE|DAE gain|Paired95% interval|
|---|---:|---:|---:|---|
|20%screen|0.759536768|0.761009521|+0.001472753|[+0.001275695,+0.001652786]|
|Full confirmation|0.755988900|0.757449150|+0.001460249|[+0.001367337,+0.001552716]|

Screen uses9,878users; full cohort49,420. User bootstrap200replicates: each user contributes exactly one event, keeping all10candidate rows together. Full confirmation includes screening users; it is not a separate untouched holdout. Bootstrap intervals condition on this recipe and frozen weighting, without adjustment for the wider research search.

## Full confirmation by arrival

|Arrival day|Events|Clicked events|Control AUC|DAE AUC|Gain|95% paired interval|
|---|---:|---:|---:|---:|---:|---|
|1|34,912|21,907|0.766431035|0.767747836|+0.001316801|[+0.001215568,+0.001418839]|
|2|8,988|5,621|0.736702468|0.738377299|+0.001674831|[+0.001453093,+0.001885776]|
|3|4,023|2,610|0.718420063|0.720579825|+0.002159761|[+0.001829708,+0.002534868]|
|4|1,497|964|0.715464026|0.717552867|+0.002088841|[+0.001418024,+0.002738174]|

All four gains exceed+0.0005. The original three-day training cutoff is1737590400; parameters and item statistics remain frozen over this horizon. Supported arrival weights are67:17:7:4 normalized over4days. Evaluation arrivals on days5–7 remain unsupported. Later-arrival AUC falls substantially for both models; different arrival cohorts contribute to this pattern, so it is not a pure causal estimate of model aging.

## History lengths

|History|Events|Clicked events|Control AUC|DAE AUC|DAE gain|
|---|---:|---:|---:|---:|---:|
|0|693|309|0.533986847|0.533890798|-0.000096049|
|1–5|9,265|5,560|0.676881123|0.680086912|+0.003205789|
|6–20|39,462|25,233|0.770231192|0.771492776|+0.001261584|

The strongest gain is for histories1–5. The zero-history group has a small full-cohort loss(-0.000096); the20%sample estimated a larger loss(-0.001205) from only160events. Do not hide this or add a post-hoc subgroup switch. Weighted history shares are1.26%/16.37%/82.37%; cohort and sample weight marginals pass independently.

## Variational comparison

VAE screen gain+0.001449260; full gain+0.001444402. VAE minus DAE on the screen is-0.000023493,95%[-4.4433221085005134e-05, -1.6974959897692042e-06]. Keep DAE as the simpler surviving arm; there is no evidence of additional benefit from this fixed variational configuration.

## What the new model learns

An unordered click-count profile is reconstructed with an exact50,000-item multinomial likelihood through a compact nonlinear bottleneck. Training profiles contain deduplicated actual pre-cutoff clicked user/item/time triples plus the earliest available supplied history for each pre-cutoff user. There are47,594nonempty profiles. Every contributing timestamp precedes the cutoff; missing warm-up exposures are never invented as observed nonclicks.

At scoring, the encoder sees the event’s supplied history, including valid after-cutoff clicks strictly earlier than that event. It outputs a deterministic mean preference vector. Candidate logits use the personal decoder dot product plus the original frozen item CTR bias; the learned catalog-marginal decoder bias is a training nuisance. Conditional10-way probabilities retain the original event-click mass. This last part is our adaptation to the actual slate setting. A fixed5percentagepoints of the old warm-component rank is replaced; ranks are joint across the complete permitted horizon, identically for both arms. No new propensity/calibration head is fitted.

The source idea is [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814), Dawen Liang, RahulG.Krishnan, MatthewD.Hoffman and TonyJebara(2018). It motivates catalog multinomial likelihood and a variational bottleneck; it does not establish the gains reported here. This compact warm-started implementation is our own adaptation, not a reproduction. No repository code or external data was copied, and no third-party code license is imported. This family was absent from earlier experiment logs; it learns an unordered catalog reconstruction objective rather than a sequential/event-softmax residual head.

## Execution and audit

[Private saved Kaggle Version2](https://www.kaggle.com/code/riazahmadbutt/techarena-codex-catalog-ae-matched-test?scriptVersionId=363749125). TeslaT4 GPU; both20epoch arms trained from scratch there. Model fitting/scoring/bootstrap took86.417s; total notebook runtime118.3s. GPU pilot estimated110.460s conservatively. Local heavy training was stopped at12DAEepochs when the user requested Kaggle, before validation scoring. Future heavy training should use Kaggle where supported.

Version1 is retained as a failed port: AST extraction accidentally omitted the original@torch.no_grad inference decorator. It stopped after DAE training before any validation labels were accessed. Version2 restores that decorator in the notebook only; all training settings, data and screening rules are unchanged. No result-driven recipe repair occurred.

Upload first encountered an automatic-review consent block. The exact25.3MBpayload, private Kaggle destination and excluded files were described; the user delegated the best option, and the subsequent private upload was accepted. Neither dataset nor notebook was made public.

Independent read-only local audit reproduced full AUC within2.7e-12(floating-point aggregation); matched frozen control/manifest; checked single-user cohort,10candidates,at-most-oneclick, strict history times, cutoff fitting, IPF marginals, finite predictions, unchanged event mass, history/candidate permutation and label invariance.29protected hashes unchanged. Saved CPU inference checks did not retrain a model. No holdout was scored, evaluation file read/scored, or submission created.

The preceding directional uncertainty diagnostic was rejected:0.759031201→0.758659198(-0.000372002). Its separate result remains in cdx_uncertainty_001/report.json; no follow-up tuning was performed.

## Artifacts and next boundary

- Frozen recipe, upload manifest and receipt:artifacts/v54_horizon/cdx_catalog_ae_001/.
- Traineddae.pt,vae.pt,frozen_predictions.npz andreport.json:cdx_catalog_ae_001/kaggle_outputs/cdx_catalog_ae_results/.
- Independent checks and full-history metrics:cdx_catalog_ae_001/postfit_audit.json.
- Protected V53/current strongest files remain unchanged. FullV53 reconstruction and current nxa_h1 integration are still untested. No submission or official-score gain is claimed. Pause after this confirmed matched-control result.
