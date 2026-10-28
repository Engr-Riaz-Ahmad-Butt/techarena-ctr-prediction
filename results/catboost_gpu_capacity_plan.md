# CatBoost GPU capacity experiment

Goal: improve the same main history-adjusted AUC toward 0.83. This experiment
is fixed before observing its results and does not claim a leaderboard score.

The preceding regularized neural candidate completed on Kaggle CUDA and failed:
day6 adjusted 0.8170778958654134, ordinary 0.8253223099529823, paired adjusted
gain interval [-0.00014157248867474901, -0.00004072825055534114]. Its initial
replay error was 2.980232238769531e-7 and saved-model replay passed. It was
preserved in Kaggle version 3, not promoted. The access-recovery follow-up was
disabled after it completed; the broader 0.83 goal remains active and unmet.

## Frozen recipe

- Original checksum-verified 354-feature bundle; no new data upload.
- CatBoost GPU, device 0, version observed in Kaggle 1.2.10.
- 1600 trees, depth 8, learning rate 0.04, L2 regularization 10.
- 63 numerical borders, Bernoulli subsample 0.8, seed 6801, Plain boosting.
- Day6 fitting days 3/4/5; day5 confirmation fitting days 2/3/4.
- All fitting timestamps must strictly precede validation cutoff.
- No validation labels in fitting, no eval_set or early stopping.
- Fixed 75% final V53 / 25% new expert rank blend.
- Positive ordinary gain and positive lower conditional adjusted-gain interval
  on day6 before unchanged day5 confirmation. Both gates required for review.
- Model and exact prediction replay checked before reporting; scratch feature
  matrix removed before making a small result archive. No CSV export.

CPU integration with synthetic data passed before launching on Kaggle. It
checks matrix alignment, scoring, model replay, scratch cleanup, gate-based
confirmation skipping and temporal-boundary rejection. It does not establish
GPU equivalence or any real score. The actual Kaggle job is now running.

Source: `kaggle/catboost_gpu_runner.py`; portable notebook:
`kaggle/TechArena_CTR_CatBoost_GPU.ipynb`.

## Weighted follow-up frozen before launch

GPU001 completed and failed the day6 gain gate (adjusted 0.8169346174; exact saved-model replay PASS). It was preserved in Kaggle Version 4. GPU002 keeps all 1600-tree parameters and 25% blend fixed, changing only fitting weights. These match the unlabeled evaluation history distribution using the existing private supplemental bundle. Per-event chronology and every one-hot click label must align with feature-day labels before fitting. No validation labels are optimized or used for stopping. Both dates remain reused development checks, not independent proof of generalization.
