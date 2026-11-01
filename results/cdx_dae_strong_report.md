# DAE stronger-baseline transfer: rejected

The fixed DAE integration decreases AUC on both20%user screens against the saved strongest H1/V53 control. The gain found with the earlier simplified horizon control does not transfer through this frozen integration recipe. No full-cohort candidate metrics, holdout, evaluation scoring, model promotion or submission were performed.

|Day|Users in screen|History weighting|Frozen control AUC|DAE AUC|Difference|Paired95% interval|
|---|---:|---|---:|---:|---:|---|
|5|7,017|3bin|0.816673662|0.816006248|-0.000667414|[-0.000866085,-0.000485384]|
|5|7,017|5bin|0.814626858|0.813973405|-0.000653452|[-0.000860802,-0.000466703]|
|6|4,896|3bin|0.821542685|0.820965333|-0.000577352|[-0.000812579,-0.000365666]|
|6|4,896|5bin|0.819305255|0.818741797|-0.000563458|[-0.000803078,-0.000361600]|

All four primary intervals are below zero. The+0.0002screen gate fails on both dates and under both weighting definitions. The+0.0005full-confirmation stage was therefore not opened. No alternative blend weights, reversed corrections, seeds, epochs or subgroup switches were tried.

The3-bin weights are the supplied evaluation proportions for histories0/1–5/6–20:1.26%/16.37%/82.37%, fitted without labels separately for each daily screen. The5-bin weights retain the existing project comparison. **Day6DAE AUC0.820965 under3-bin weighting is a regression from its0.821543control. It is not evidence of a new0.82gain.** These screen scores must not be compared directly with the previously reported full-day5-bin0.818023.

Bootstrap uncertainty uses200paired replicates of users, with all10candidate rows retained together. Each allowed daily cohort contains exactly one event per user. The dates have been reused throughout the project; these are conditional screening intervals, not an independent official test.

The same fixed correction also loses against both secondary parents:

|Parent|Day|3-bin difference|5-bin difference|
|---|---:|---:|---:|
|H1 with V39 parent|5|-0.000691172|-0.000679070|
|H1 with V39 parent|6|-0.000602955|-0.000590251|
|Unheaded V53|5|-0.000543104|-0.000531691|
|Unheaded V53|6|-0.000444391|-0.000434500|

This was one already-surviving DAE recipe, refitted at the proper day5/day6cutoffs:16dimensions,20fixed epochs,50%input corruption,Adam0.001,batch256,seed8711. Initial history/item factors come from the existing same-day warm16seed7checkpoints. Fit profiles contain deduplicated actual pre-cutoff clicked user/item/time triples plus each pre-cutoff user’s earliest supplied history. Missing warm-up exposures are not invented as observed nonclicks. There is no new stacker, calibration model or event-propensity fit.

|Scored day|Fit cutoff|Latest fit timestamp|Training events|Nonempty fit profiles|Allowed cohort|20%screen|
|---|---:|---:|---:|---:|---:|---:|
|5|1737763200|1737763140|490,420|51,044|35,230|7,017|
|6|1737849600|1737849540|588,873|51,904|24,422|4,896|

At inference, the same learned unordered encoder uses supplied event history. The personal decoder dot is combined with the original item CTR bias; original warm-model event-click mass is preserved. The scoring displacement is fixed at5percentagepoints:rank(parent)+0.05×[rank(DAE)-rank(original warm16)]. Ranks use the complete allowed daily cohort and do not depend on labels; sample-only or per-segment ranks are not used. The daily pipelines have separate cutoffs. This validation recipe is a score diagnostic; a stateless calibrated evaluation export was not built.

The saved day5H1reference inherits2-fold user cross-fitting within day5. Day6H1was fitted on earlier day5. Those parent heads were read-only and were not refitted. New DAE training is strictly pre-cutoff on both dates; this test does not establish a new fully causal reconstruction of the entire V53 pipeline.

[Private Kaggle Version1](https://www.kaggle.com/code/riazahmadbutt/techarena-codex-dae-strong-baseline?scriptVersionId=363765920) used TeslaT4. Fitting, scoring and assessment took64.523seconds; total notebook runtime106.5seconds. The first-epoch timing pilots estimated169.5/127.3seconds conservatively. Setup and browser/download recovery took additional time. All heavy fits ran on Kaggle through the in-app browser.

Independent read-only audit passed: manifest and all frozen parent hashes match;20fixed epochs; exact stored prediction formula; allowed event alignment and deterministic user sampling; strict-past histories; unchanged event-click mass; sample AUC replay; label independence and candidate-order consistency. CPU replay of the saved Kaggle checkpoints differs from GPU probabilities by at most3.15e-7/2.09e-7 for128events per date. This replay performed no training.29protected files remain unchanged.

Both checkpoints and all allowed daily predictions were frozen before screening outcomes were accessed. Only the20%screen outcomes were assessed after the gate failed. The bundle and outputs contain no holdout or evaluation data, and no submission file was created or modified.

The earlier cut3horizon DAE result remains a positive result against its simplified control: full weighted AUC+0.001460. This stronger-parent result prevents promoting that earlier gain into a claim about the main model. The current direct integration recipe is rejected. The strongest original model and official-score ledger remain unchanged; an official0.82score is unproven.

[Independent audit](C:/Users/engrr/Documents/ChatGPT/Task/techarena-ctr-prediction/artifacts/v54_horizon/cdx_dae_strong_001/postfit_audit.json) · [Raw Kaggle report](C:/Users/engrr/Documents/ChatGPT/Task/techarena-ctr-prediction/artifacts/v54_horizon/cdx_dae_strong_001/kaggle_outputs/cdx_dae_strong_results/report.json) · [Frozen recipe](C:/Users/engrr/Documents/ChatGPT/Task/techarena-ctr-prediction/artifacts/v54_horizon/cdx_dae_strong_001/frozen.json)

![Kaggle screening table](C:/Users/engrr/Documents/ChatGPT/Task/techarena-ctr-prediction/results/cdx_dae_strong_screen.png)
