# Kaggle joint embedding experiments

The requested objective is 0.83. It remains unmet. V53 remains the stored
accepted baseline. The strongest recent same-cohort candidate is
ctr-main-003-interactions: day6 adjusted AUC 0.8172634369364146, ordinary
AUC 0.8254950475978616. Its day5 gate and exact checkpoint replay passed.
Confirmed official score: V35, 0.816623. No newer official score is claimed.

## Prepared data

The private supplementary Kaggle dataset is:
https://www.kaggle.com/datasets/riazahmadbutt/techarena-neural-causal-histories

Created and attached to the existing notebook through the authorized UI.
The 108461582-byte source archive contains 12 files and has SHA256
f5000323fa682a192577624a74b93be16addf9bc428d605ea81c09ef1843f9fd.
No original large CSV re-upload was needed. Event IDs and user/item IDs are
encoded categorical indices; no semantic arithmetic is applied to opaque IDs.

Histories have strictly earlier timestamps. Training event indices exclude
validation indices and precede their dates. Initial warm factors were computed
before day-2; the audited V36 context checkpoint uses only preceding dates.
Initial model predictions reproduce the existing V36 checkpoint within 3e-6.
The source builder checks original row labels and event order. Both dates
have repeatedly been reused, so reported intervals exclude selection effects.

## ctr-neural-001: rejected

Six fixed epochs on the preceding two days, all warm embeddings trainable.
Fixed 10% rank-residual replacement against the original neural component.

- Day6 adjusted AUC: 0.8171415703710364.
- Ordinary AUC: 0.8253478591701169.
- Raw neural AUC increased from 0.7777203183 to 0.7911178062, but the combined
  predictor declined. This is not a successful candidate.
- Initial complete-cohort replay error: 2.980232238769531e-7.
- Saved checkpoint prediction replay passed. No day5 fit or CSV export.
- Kaggle results archive for this rejected run was created in the working directory.

## ctr-neural-002-aux64: rejected

64 dimensions, twenty fixed epochs, all 588873 events before day6 and 575915
deduplicated supplied warm-click examples. Sampled softmax auxiliary loss with
ten negatives and popularity correction. No held-out labels were optimized.

- Day6 adjusted AUC: 0.8149403388523853.
- Ordinary AUC: 0.8229931805683433.
- Raw AUC: 0.6746618179578381.
- Training event loss fell from 1.269095 to 0.0145905 while validation ranking
  deteriorated: the large low-regularization model overfit severely.
- Full initial replay and exact saved-checkpoint replay passed.
- No day5 fit, promotion, or competition CSV.

## ctr-neural-003-regularized: prepared, not run

The next independent recipe returns to 16 dimensions, six fixed epochs and
learning rate 0.0003. It penalizes sampled user/item/history vector drift with
coefficient 0.2, rather than averaging a small anchor term over entire embedding
tables. User dropout is 0.25; history dropout is 0.1. Inference has no dropout.
It retains full-prefix fitting, the warm-click objective and fixed 10% residual.

Notebook: kaggle/TechArena_CTR_Neural_Regularized.ipynb.
The module is embedded in the notebook with an exact source checksum, then
copied into the output archive. Input files remain checksum verified.

Automatic browser approval review could not authorize the subsequent Kaggle
action because the browser tool usage limit was hit. The regularized run has
not started. No browser workaround or indirect Kaggle execution was attempted.
The user requested that the assistant resume once authorized access recovers.
An hourly follow-up was created for this chat; it stays quiet while access is
unchanged and disables itself once this prepared run is completed and reported.

## Verification

Local invariant checks cover candidate permutation symmetry, the no-click
option, empty-history masking, nonzero embedding gradients, capacity expansion,
exclusion of non-fitting events from auxiliary pairs, inference equivalence with
dropout disabled and state-dictionary replay. Source and notebook cells compile.
These checks establish implementation behavior, not a higher validation score.
