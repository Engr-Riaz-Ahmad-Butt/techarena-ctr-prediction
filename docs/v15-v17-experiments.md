# V15–V17 local experiments

| Version | January 26 AUC | January 25 AUC | January 26 history-adjusted | January 25 history-adjusted |
|---|---:|---:|---:|---:|
| V14 | 0.813889863 | 0.804728376 | 0.805837112 | 0.797856273 |
| V15 | 0.815071283 | 0.805872968 | 0.806894741 | 0.798914350 |
| V16 | 0.816071456 | 0.807311312 | 0.807927808 | 0.800239786 |
| V17 | 0.816134690 | 0.807375185 | 0.807982616 | 0.800304001 |

V15 reblends existing components with the new warm16 model. V16 adds four causally generated warm-model score features to stacked CatBoost and replaces the old stack member. Each training block's factor checkpoint is trained strictly before that block. V17 averages three warm-factor training seeds for its direct warm component; this adds only a marginal gain over V16.

V17 recipe: 0.9 times a 50% event-competition adjustment of sigmoid(0.7 logit(warm-feature stack) + 0.2 logit(V6) + 0.1 logit(V7)), plus 0.1 times the mean warm16 raw predictions for seeds 151, 7 and 1009. All weights/tree counts selected on January 26 are fixed for January 25. Both dates have been repeatedly reused; they are not independent untouched tests.

Recorded independent checks verify recipes, row coverage, finite probabilities and matching event/label arrays. The actual V17 file SHA-256 is bb4b1e5521f80733c7f97c4698d8245216cfda886faf21a131f27f17d0f6e31b, matching the tested candidate. File aliases may differ from historical report names after promotion. The official V17 score is unknown.

## Reproduction order

Reproduce V14 and earlier artifacts first. The explicit CTR_BASELINE_VERSION variable selects the comparison baseline, avoiding a circular dependency when rebuilding earlier experiments. This is a packaging/path change; stored scores are historical experiment results.

```powershell
$env:CTR_BASELINE_VERSION='V14'
python imp3_reblend.py --tag reblend
$env:CTR_BASELINE_VERSION='V15'
python imp3_warmfeat.py --power 0
python imp3_reblend.py --tag wf --override stack=warmfeat
python imp3_warm_seeds.py --seeds 7 1009
$env:CTR_BASELINE_VERSION='V16'
python imp3_reblend.py --tag wf3 --override stack=warmfeat --warm-seeds 151,7,1009
$env:CTR_BASELINE_VERSION='V17'
```

The advanced full pipeline takes substantial CPU time. Data, features, checkpoints and CSV submissions are not committed. Reproduction writes candidate files; only promote one after coverage, recipe and metric checks.

## Rejected variants

Importance-weighted training, a short-history specialist and post-hoc transforms did not improve both dates. Pairwise ranking was worse than the classification stack (0.801992/0.776421). These are documented failed experiments, not newer recommended submissions. See results/imp3_*_report.json.
