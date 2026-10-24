# Local AUC target: 0.84

The target has not been reached. This is a local validation target, not an
official competition score prediction.

The independently accepted V20 baseline is 0.8215282579633503 on January 26
and 0.814599936313616 on January 25. Its submission CSV is preserved.

The saved frozen-ALS DIN epoch-six expert, blended at 20% using the frozen
rank recipe, scores 0.8237331494250395 on the January 26 cohort. Independent
cohort, chronology and sampled checkpoint replay checks passed. This result
is provisional until the untouched January 25 confirmation completes.

The confirmation is running with exactly six epochs, ALS cutoff three,
learning rate 0.003, 64 dimensions, seed 42, click-only training and frozen
item embeddings. Four CPU threads; one training job started by this run.
Outputs use the separate prefix `artifacts/v21_als6_confirm_day5`.
Do not restart while that job is active or overwrite its checkpoints.

The fixed-epoch path passed an isolated fixture where validation AUC fell
every epoch: it still ran all six epochs and retained epoch six. Existing
model outputs are rejected; an existing log file is permitted.

After successful training completion:

```powershell
python v21_confirm_blend.py
python v21_blend_forms.py --confirm
```

The second script evaluated 12 additional primary-date blend settings;
none beat the 20% rank blend. The search is exploratory and its uncertainty
is not included in conditional paired intervals. Neither script exports a
submission or certifies 0.84. Require independent both-date metric and
recipe audits before promotion, followed by an exact evaluation refit.

Frozen input/config manifest: `artifacts/v21_als6_blend_frozen.json`.
Frozen source hashes: `artifacts/v21_als6_source_frozen.json`.

The separate experimental pipeline now requires both-date gains over V20
before retaining a stage, uses fixed rounds on January 25, and writes an
exclusive separate candidate filename. Its unconfirmed multi-model stage
cannot be promoted based on an LGB-only confirmation.
