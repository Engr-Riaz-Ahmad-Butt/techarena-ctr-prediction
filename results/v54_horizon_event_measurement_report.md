# Idea2 — event-click measurement only

Local runtime: 29.1 seconds. Event model fit on 43,999 first-user events after the day1 cutoff, with labels strictly before day3. Frozen validation:49,420 first-user events after day3. One fixed shallow LightGBM; no tuning against validation outcomes.

|Arrival|Events|Clicked events|Weighted AUC|95% user-bootstrap interval|
|---|---:|---:|---:|---|
|overall|49,420|31,102|0.627487|0.622343–0.632396|
|arrival1|34,912|21,907|0.631227|0.624762–0.637176|
|arrival2|8,988|5,621|0.622627|0.611660–0.635065|
|arrival3|4,023|2,610|0.619314|0.601819–0.639406|
|arrival4|1,497|964|0.596920|0.569243–0.627624|
|arrival3plus|5,520|3,574|0.611276|0.594811–0.626092|
|screen20_overall|9,878|6,206|0.622135|0.610835–0.632827|

400 user bootstraps, fixed fitted model and fixed IPF weights. Arrival targets67:17:7:4 normalized over four days; history targets1.26%/16.37%/82.37%. Evaluation days5–7 unsupported. Validation label sentinels and counterfactual-label check passed;29 protected artifacts checksum unchanged. No holdout opened or scored. No submission.

Event-containing-click share: 62.934035% unweighted, 63.095878% weighted. The simplified blend maximum predicts event clicks with AUC0.584023 versus0.627487 for the small event model (paired delta+0.043464,95% interval+0.037931 to+0.049091). This comparison concerns EVENT labels; it is not an item-AUC gain.

V53 already includes pre-cutoff user click/event counts, smoothed user propensity and interactions, recent rates, user-bucket propensity, and event/history summaries. Saved selection-day V53 maximum-score event AUCs:day5=0.621727,day6=0.623142. Event click share rises from48.35% to75.83% across day5 score quintiles, and47.42% to76.11% on day6. These are different cohorts from horizon validation; no direct incremental gain claimed.

Measurement gate passes. Pause before any probability-product or item-choice construction. Historical fit supports two arrival days; validation has four, so long-horizon results involve extrapolation. Native event probabilities are not certified calibrated. Full V53 horizon reconstruction remains pending.

Kaggle rerun requested: portable runner retrains the identical event classifier using the frozen causal features; no upstream factor model refitting. Bundle excludes evaluation data, reserved holdout, original V53 files, submissions, and credentials. Cloud execution status tracked separately.

## Kaggle execution verified

[Private saved notebook](https://www.kaggle.com/code/riazahmadbutt/techarena-idea2-event-click-measurement), version1: Event measurement only - verified on Kaggle. Fixed event classifier retrained on frozen causal features; upstream factors were prepared locally. Python3.13.15,LightGBM4.6.0. Model/measurement runtime6.138s; complete first-cell wall time13.509s. Weighted AUC0.627487218457; maximum prediction difference from local1.11e-16. Input manifest checks passed on Kaggle. Saved version includes output. CPU session stopped. Protected29 hashes rechecked unchanged. Measurement stage complete; pause before probability-product model.
