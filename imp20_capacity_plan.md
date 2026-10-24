# Larger tree capacity on the causal ALS feature set

October 24 update: target local AUC is now 0.84. Run this experiment only after
the active fixed-six-epoch DIN confirmation finishes. Keep V20 as the locked,
independently accepted baseline; the 0.823733 DIN blend is still provisional.
Use four CPU threads instead of eight to reduce laptop load. This is the only
operational change in addition to the 31-to-63-leaf model change below. A gain
over V20 alone does not establish a gain over the current best DIN blend.

Prepared before any imp20 fit or score. The accepted imp19/V20 baseline is locked by both-date prediction hashes and its independent acceptance audit. Retain the full January26 first-event cohort and frozen-setting January25 confirmation.

Hypothesis: the 354 existing causal features include many aligned factor-component interactions. A 31-leaf learner may constrain their joint representation. Increase num_leaves from31 to63 and reduce CPU threads from8 to4. Retain learning rate.04, min_data_in_leaf200, lambda_l2=2, feature_fraction.9, bagging_fraction.8/freq1, max_bin127, seed42 and objectivebinary/metricauc. No new data/features or changed validation population.

Train the same rolling all-event blocks3/4/5 for January26 and2/3/4 for January25, 354features. Maximum1600 rounds with ordinary-AUC early stopping120 on January26. Try rank blends .1/.25/.5/.75/1 with the accepted incumbent. Select highest ordinary AUC only among ordinary+history-weighted gains; require positive conditional paired intervals before confirmation. Freeze trees/weight for January25. Promote only after both-date gains and independent native-model/full-recipe/metric/prefix audits. Save raw model predictions even if rejected. Reused-validation selection uncertainty remains. Ranking-only experiment, no CSV claim, one heavy fit at a time.
