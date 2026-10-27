# Corrected sequence residual: rejected

V53 remains the best audited local candidate: history-adjusted day6 AUC 0.8171690462350286. The user requires at least 0.82 adjusted AUC before another upload recommendation. The goal 0.834786 remains unmet; official best is still user-confirmed V35 = 0.816623. No CSV was exported.

V57 uses the regression-tested right-padded history helper, encodes each history once per event, masks empty-history residuals, and trains an 11-way loss including no-click. Warm factors and the V36 context model stay frozen. A 32-unit GRU consumes fixed history/item embeddings and log time gaps; its zero-initialized head produces an interest-vector correction and a common click-logit correction. Metadata is prepared in small blocks. One fitting process uses four Torch threads.

The recipe was frozen before fitting: Adam learning rate 0.001, weight decay 0.0001, residual-square penalty 0.01, dropout 0.1, batch size 512. Day6 selects among epochs 1/2/3 and rank-residual weights 0.05/0.1. The same choice would be confirmed on day5 only after a positive ordinary gain and positive adjusted lower interval on day6. Fitting labels strictly precede the validation cutoff. The initial zero-residual predictions reproduced V36 exactly.

An initial run failed before training because one-hot encoding required an int64 bucket tensor. The repaired driver explicitly converts that tensor, then writes a new frozen manifest. The failed manifest and traceback remain in ignored artifacts. No training settings or validation scores guided this dtype repair.

The best of six fixed day6 settings was epoch3, weight0.05:

| Metric | V53 | V57 |
|---|---:|---:|
| Ordinary AUC | 0.8254233318844953 | 0.8254086261352410 |
| History-adjusted AUC | 0.8171690462350286 | 0.8171577192448297 |

Adjusted difference: -0.000011326990198901044; conditional event-influence interval [-0.000029259730795562264, 0.000006605750397760177]. The candidate failed the gate, so no day5 confirmation was fitted. Declining fitting loss did not imply improved validation ranking.

Independent saved-checkpoint replay covered all 351,380 main validation rows. Maximum raw replay error was 1.1921e-7; frozen-context replay error 5.9605e-8. The final rank scores and AUCs reproduced exactly. Frozen source/input hashes, selected settings, context checkpoint tensors, history ordering, causal history timestamps, labels/events and empty-history probability equality passed. Reports retain ordinary and adjusted scores separately. Validation dates have been repeatedly reused; intervals exclude fitting/selection uncertainty and do not forecast an official score.

The legacy packing bug is real, but correcting it did not produce a stronger candidate in this residual experiment. This result does not establish that every recurrent architecture fails; it does rule out promoting this tested recipe.
