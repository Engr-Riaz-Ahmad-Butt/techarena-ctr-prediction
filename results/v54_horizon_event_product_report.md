# Idea2 Step2 — two-part probability matched control

The event factor improves conditional-choice-only scores and adds a smaller signal beyond score summaries. It does not establish an improvement over untouched V53. The saved day6 V53 score remains better than every tested two-part replacement. Keep V53 unchanged and pause.

Compute on Kaggle CPU: 157.02s for the fixed logistic/boosted fits, earlier-block calibration and primary bootstrap report. A further reporting-only cell bootstrapped differences against unchanged original scores; no refits. The frozen horizon cohort and weights were preserved. No holdout or evaluation labels, external data, submissions, or protected writes.29 protected artifacts passed checksum comparison.

## What was fitted

The horizon uses the previously authorized simplified blend of existing V53 contributors:95% warm16 and5% ALS. The complete V53 horizon rebuild is absent. Each raw expert score is mapped with an empirical CDF fitted only on the earlier historical training partition. No validation or evaluation-file score distribution is used to fit these CDFs. A single positive inverse temperature is fitted by conditional clicked-item cross-entropy on later pre-validation events. This is a monotone scalar mapping followed by softmax over exactly ten items, producing conditional probabilities summing to one.

The inverse temperature is 2.546210699 for the horizon and 4.801510288 for the saved V53 diagnostic. Calibration uses 5,723 clicked events in the horizon historical calibration block.

Historical horizon event data: 43,999 first-user events after cutoff1 and strictly before cutoff3. A temporary event model fits the earliest 35,156 events and scores the later 8,843; positive-slope Platt calibration is fitted on those temporally out-of-sample probabilities. The final event models fit all43,999 permitted historical events and score the49,420 frozen first-user validation events. The boosted basic arm reuses the exact measurement model, followed by the earlier-block Platt map.

The simple logistic uses five inputs: activity gap, hours since cutoff, history length, shrunk pre-cutoff user event-click rate, and mean ALS score. The small boosted version uses the ten measurement features and the fixed120-tree,7-leaf,depth3 recipe. No settings were searched.

Every matched arm shares the same conditional item probability q. The choice-only control is prior_click_rate*q; its constant factor has no effect on AUC. The candidate is calibrated_event_probability*q. For the stronger comparison, the control event model uses max/mean item score, user click rate and history length; the augmented model adds the full measurement features. Logistic and boosted strengthened comparisons each retain their own matched control.

## Global weighted row AUC

|Cohort|Comparison|Control|Candidate|Gain|Paired95% interval|
|---|---|---:|---:|---:|---|
|Horizon|logistic_vs_choice|0.752211990|0.755326063|+0.003114073|+0.002626556 to +0.003533568|
|Horizon|boosted_vs_choice|0.752211990|0.755036395|+0.002824405|+0.002348530 to +0.003304004|
|Horizon|logistic_beyond_existing|0.754645383|0.755554457|+0.000909074|+0.000676759 to +0.001152789|
|Horizon|boosted_beyond_existing|0.754461954|0.755173224|+0.000711270|+0.000510908 to +0.000892471|
|Saved V53 day6|logistic_vs_choice|0.814831678|0.818455959|+0.003624281|+0.002964340 to +0.004189870|
|Saved V53 day6|boosted_vs_choice|0.814831678|0.818278967|+0.003447289|+0.002835445 to +0.004019106|
|Saved V53 day6|logistic_beyond_existing|0.818360421|0.818817715|+0.000457294|+0.000230246 to +0.000662849|
|Saved V53 day6|boosted_beyond_existing|0.818310652|0.818633973|+0.000323320|+0.000191276 to +0.000457732|

All four full-cohort matched comparisons clear the requested overall+0.0002 point threshold. On the20% horizon sample, boosted-beyond-existing gains only+0.000144 and fails the screening threshold; logistic-beyond-existing gains+0.000660. Full-cohort results are reported for the fixed requested models, including this screening failure; they are not a promotion of a failed screen.

## Comparison against the original scores

|Cohort|Arm|Original AUC|Candidate AUC|Gain|Paired95% interval|
|---|---|---:|---:|---:|---|
|Horizon simplified blend|logistic_basic|0.755280328|0.755326063|+0.000045735|-0.000513512 to +0.000527302|
|Horizon simplified blend|boosted_basic|0.755280328|0.755036395|-0.000243934|-0.000792384 to +0.000275471|
|Horizon simplified blend|logistic_augmented|0.755280328|0.755554457|+0.000274129|-0.000246786 to +0.000751823|
|Horizon simplified blend|boosted_augmented|0.755280328|0.755173224|-0.000107104|-0.000620778 to +0.000429424|
|Saved full V53 day6|logistic_basic|0.820037962|0.818455959|-0.001582003|-0.001953029 to -0.001192313|
|Saved full V53 day6|boosted_basic|0.820037962|0.818278967|-0.001758995|-0.002129430 to -0.001431430|
|Saved full V53 day6|logistic_augmented|0.820037962|0.818817715|-0.001220247|-0.001552856 to -0.000877860|
|Saved full V53 day6|boosted_augmented|0.820037962|0.818633973|-0.001403989|-0.001716730 to -0.001103424|

The day6 untouched V53 weighted AUC is0.820037962. Its strongest two-part replacement, logistic-augmented, reaches0.818817715: a loss of0.001220247. Thus positive gains over conditional-choice or reduced summary controls must not be described as a gain on top of V53. On the horizon, logistic-augmented is only+0.000274 over the original simplified blend, and its20% sample score is below that original baseline.

## Horizon arrival and history groups

Arrival and history bins below overlap; do not sum them. Weights match normalized arrival67:17:7:4 and history1.26%/16.37%/82.37%, preserving the observed association. Evaluation arrivaldays5–7 are unsupported. Full-cohort uncertainty uses200 paired user bootstraps; screening uses400. Model, calibration and IPF weights stay fixed during resampling.

### logistic_vs_choice

|Group|Events|Clicks|ESS|Control AUC|Candidate AUC|Gain|Paired95% interval|
|---|---:|---:|---:|---:|---:|---:|---|
|overall|49,420|31,102|48907.6|0.752211990|0.755326063|+0.003114073|+0.002626556 to +0.003533568|
|arrival1|34,912|21,907|34784.1|0.762409388|0.765543955|+0.003134567|+0.002648791 to +0.003648411|
|arrival2|8,988|5,621|8920.5|0.733118355|0.736460948|+0.003342593|+0.002408416 to +0.004210007|
|arrival3plus|5,520|3,574|5242.3|0.715385210|0.717452148|+0.002066938|+0.001031233 to +0.003174359|
|history0|693|309|675.7|0.517799192|0.525138145|+0.007338953|-0.005122005 to +0.019480435|
|history1to5|9,265|5,560|9103.8|0.672092508|0.676054293|+0.003961785|+0.002666429 to +0.005360124|
|history6to20|39,462|25,233|39290.8|0.766389430|0.769531224|+0.003141794|+0.002681962 to +0.003603672|

### boosted_vs_choice

|Group|Events|Clicks|ESS|Control AUC|Candidate AUC|Gain|Paired95% interval|
|---|---:|---:|---:|---:|---:|---:|---|
|overall|49,420|31,102|48907.6|0.752211990|0.755036395|+0.002824405|+0.002348530 to +0.003304004|
|arrival1|34,912|21,907|34784.1|0.762409388|0.765146088|+0.002736700|+0.002191326 to +0.003293373|
|arrival2|8,988|5,621|8920.5|0.733118355|0.736548003|+0.003429648|+0.002421334 to +0.004498345|
|arrival3plus|5,520|3,574|5242.3|0.715385210|0.717534908|+0.002149697|+0.001014307 to +0.003303223|
|history0|693|309|675.7|0.517799192|0.527017844|+0.009218653|-0.002661097 to +0.021731712|
|history1to5|9,265|5,560|9103.8|0.672092508|0.675985684|+0.003893176|+0.002642618 to +0.005154998|
|history6to20|39,462|25,233|39290.8|0.766389430|0.769097332|+0.002707902|+0.002234327 to +0.003196666|

### logistic_beyond_existing

|Group|Events|Clicks|ESS|Control AUC|Candidate AUC|Gain|Paired95% interval|
|---|---:|---:|---:|---:|---:|---:|---|
|overall|49,420|31,102|48907.6|0.754645383|0.755554457|+0.000909074|+0.000676759 to +0.001152789|
|arrival1|34,912|21,907|34784.1|0.765057502|0.765806585|+0.000749083|+0.000515080 to +0.001048386|
|arrival2|8,988|5,621|8920.5|0.735657568|0.736727368|+0.001069800|+0.000584260 to +0.001546031|
|arrival3plus|5,520|3,574|5242.3|0.717040157|0.717390224|+0.000350068|-0.000451247 to +0.001002453|
|history0|693|309|675.7|0.529528270|0.520733160|-0.008795110|-0.016587870 to -0.001133254|
|history1to5|9,265|5,560|9103.8|0.674564280|0.676068267|+0.001503987|+0.000724100 to +0.002299576|
|history6to20|39,462|25,233|39290.8|0.768883160|0.769724359|+0.000841199|+0.000640406 to +0.001080766|

### boosted_beyond_existing

|Group|Events|Clicks|ESS|Control AUC|Candidate AUC|Gain|Paired95% interval|
|---|---:|---:|---:|---:|---:|---:|---|
|overall|49,420|31,102|48907.6|0.754461954|0.755173224|+0.000711270|+0.000510908 to +0.000892471|
|arrival1|34,912|21,907|34784.1|0.764795100|0.765387378|+0.000592279|+0.000373587 to +0.000876742|
|arrival2|8,988|5,621|8920.5|0.735850587|0.736441144|+0.000590557|+0.000094781 to +0.000957305|
|arrival3plus|5,520|3,574|5242.3|0.716836393|0.717462185|+0.000625792|+0.000089676 to +0.001179935|
|history0|693|309|675.7|0.524963967|0.526986116|+0.002022148|-0.003320050 to +0.007495561|
|history1to5|9,265|5,560|9103.8|0.675031065|0.675932489|+0.000901424|+0.000386219 to +0.001468499|
|history6to20|39,462|25,233|39290.8|0.768597564|0.769268077|+0.000670513|+0.000482369 to +0.000880545|

Zero-history results are unstable with693 users. Logistic-augmented loses0.008795 versus its strengthened zero-history control; its interval excludes zero. The later-arrival3+ logistic-augmented gain is0.000350, but the interval crosses zero. No claim of a reliable later-arrival or zero-history improvement.

## Saved V53 diagnostic: interpretation

This is a separate day6 selection cohort of24,422 users, with history-only weighting to the same three target proportions. Its models fit35,230 day5 events. Earlier14,118 events produce out-of-sample predictions for the later21,112 pre-day6 calibration events. Final event fits and calibrators use only information available before scored day6 events. Conditional item calibration uses the saved day5 V53 scores and labels, then freezes for day6.

The day5 V53 scores are user-cross-fitted, not generated by a strictly chronological day5 head. This historical calibration input inherits original V53 ranking and fitting conventions. It is a supplemental pre-day6-available-data diagnostic, not a strict horizon reproduction, not an additional independent day5 confirmation, and not evidence that the legacy full V53 can already score without cohort statistics.

## Probability and causality checks

Horizon fitted labels end at1737590340, before the first scored validation event1737590400. Historical temporary-model fit events precede every calibration event. Saved-V53 final fitted labels end at1737849540, before day6 scoring begins1737849600. Histories are strictly earlier than their events. Prefix feature aggregates use only earlier training data; no validation-label updates occur. Frozen scoring rows contain label sentinels. All final predictions are saved before validation labels are used in the reporting routine. Changing those validation labels cannot change the saved predictions or fitted parameters.

Every output is finite and in[0,1]. The ten conditional probabilities sum to1; the ten final probabilities sum to the calibrated event-click probability, which is at most1. There is no final ranking or per-day normalization. Calibration was fitted earlier; it is not perfect after distribution shift.

|Horizon arm|Event AUC|Event Brier|Row Brier|Row logloss|Predicted click-event share|Observed weighted share|
|---|---:|---:|---:|---:|---:|---:|
|logistic_basic|0.625880|0.222503|0.055359|0.209835|64.842%|63.096%|
|logistic_existing|0.597878|0.227124|0.055428|0.210232|66.547%|63.096%|
|logistic_augmented|0.628927|0.221968|0.055348|0.209780|64.678%|63.096%|
|boosted_basic|0.627487|0.222183|0.055352|0.209801|63.953%|63.096%|
|boosted_existing|0.613894|0.224515|0.055345|0.209961|65.637%|63.096%|
|boosted_augmented|0.630327|0.221742|0.055319|0.209737|64.429%|63.096%|

For example, logistic-augmented predicts64.68% click-containing events versus63.10% observed under the frozen weights. No correction was fitted on these validation outcomes. These are earlier-calibrated coherent probabilities, not a claim of perfect calibration on the shifted target cohort.

## Exact potential evaluation application — not executed

The tested replacement is not recommended for evaluation because it underperforms existing full V53. If a later approved confirmation supports application, the stateless procedure is:

1. From training data strictly before the real cutoff, fit the upstream item/user factors, user statistics, historical score-CDF reference arrays, conditional softmax temperature and event model/calibrator using chronological out-of-sample blocks. Freeze them all.
2. For each evaluation event, use its user ID, candidate IDs, timestamp, context and supplied clicks strictly before that event. Compute user rates from frozen training counts, activity gap from prior known activity, and ALS summaries over just those ten candidates. Do not update anything from unknown evaluation outcomes.
3. Compute raw item-expert scores. Apply the frozen historical CDFs to those scores, the fixed95/5 blend, and the earlier-fitted softmax temperature. This yields q_i for the ten candidates.
4. Compute the frozen event model probability, followed by its earlier-fitted Platt mapping. Return p_event*q_i for each row in original input order. Processing one event or the whole evaluation file must produce identical probabilities.
5. Do not fit any transform, select a temperature, estimate a global score distribution, reweight training, or normalize ranks from evaluation.csv. Candidate max/mean within the current ten-item event are permitted prediction-time features.
6. Applying the same concept to full V53 would additionally require replacing its legacy rank operations with frozen historical CDFs throughout its dependencies and validating that exact pipeline. That reconstruction has not been performed here.

All bootstrap intervals condition on reused validation cohorts and fitted pipelines. They omit model-fit uncertainty and have no adjustment for the fixed multiple comparisons. No official-score improvement is claimed. Pause after these results.