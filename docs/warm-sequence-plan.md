# Contextual warm-sequence factor model: local0.83 goal

This is a new hypothesis prepared before any imp7 validation results. Existing warm auxiliary fitting supervises U/V from warm positives but does not directly teach H from warm click sequences. Add pre-target warm-history context and item bucket/daytype biases to the factor objectives.

Warm auxiliary logits: fixed smoothed item-popularity offset + item bucket bias + item daytype bias + (U + sumH/sqrt(historylength))dotV. No outside/no-click option and no event bu/bi in the warm loss. Event logits add the existing -1.8+bu+bi and retain the outside/no-click option of logit0. Context histories strictly precede their warm target timestamp; same-minute ties are excluded. Exact warm(user,item,timestamp)duplicates are removed. Real repeated items at different times are retained.

Fixed16dimensions, seed151, LR.045, latent/biasreg.002, item-contextreg.01, warm weight1.0. Initial U/V/H/bu/bi reuse the same historical SVD recipe; new context biases start0. All factor checkpoints and warm targets precede their scoring cutoff. Item and history repeated IDs aggregate gradients once per distinct ID; regularization is per occurrence. Accumulate every user/history gradient from pre-update item vectors. Save all parameters, AdaGrad accumulators and RNG state.

Select epoch from2,4,6,8,10 and probability replacement weight from0,.1,.2,.3,.5,.75,1 onJanuary26 ORDINARY global ROC AUC. This selection metric is set before fitting in response to the user's explicit clarification that LOCAL0.83 is the target; historical experiments prioritized history-weighted AUC as a leaderboard proxy. The unchanged first-event validation population is retained. Eligible settings must also improve history-weighted AUC against the explicit frozen baseline prefix. Freeze epoch/weight forJanuary25. The baseline prefix will be chosen from independently confirmed candidates BEFORE fitting begins; do not retune it after observing imp7 validation. Require independent replay and both-date improvement before final refit. Local0.83, not public AUC and not a different easier cohort, remains the user goal. Reused-validation selection uncertainty remains.

Do not run this heavy fit concurrently with the live warm-context CatBoost experiment. Independently check tiny-example gradients and chronology before fitting.

Baseline frozen before the first imp7 fit: imp5_warmcontext, after independent both-date audit PASS (January26 .8180674151802085; January25 .8107281114951901). The confirmation gain is small and its interval includes zero.
