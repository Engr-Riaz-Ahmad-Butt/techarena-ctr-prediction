# V68 fixed ALS — Kaggle exploratory result

## Status

Executed once on Kaggle after freezing the recipe in
`kaggle/v68_fixed_als_runner.py` and publishing the private code input dataset
version 5. The notebook was saved as Quick Save version 8
(`V68 exploratory day6 gated`); the save completed successfully. The original
saved notebook version 7 was not overwritten.

**Output persistence blocker:** version 8's Output tab reports 0 files. Quick
Save preserved the notebook but did not commit the run's `/kaggle/working`
artifacts. After returning to the editor, the working-directory browser listed
no files and its Jupyter proxy connection was unstable. The files and hashes
below were observed during the live Kaggle run, but the raw prediction arrays
and model checkpoint are not available in the saved version. No second training
run or submission was made. Persisting the binary artifacts requires either
recovery of that live kernel before reset or authorization for an exact
no-sweep rerun in a V68-only notebook saved with “Save & Run All.”

**Metric validity correction:** an independent evaluator found that the exact
day-6 initializer, `imp15_model_day6.npz`, had been selected conditionally on
day-6 AUC (`imp15_als_trials_day6.json` / `imp15_als.py`). Consequently, the
day-6 comparison below is label-informed and must not be interpreted as an
unbiased validation result or used for candidate comparison. The displayed
numbers are retained solely as a record of the exploratory run. No further
V68 fit was launched after this finding.

This is exploratory only. Day 5/day 6 are reused development cohorts, so these
metrics do not support tuning, promotion, or an official-score claim. No
competition submission was created. No real-data model training was run
locally.

## Independent evaluation

The runner's own fit uses pre-cutoff events and its checks show no direct
day-6-label fitting. However, the supplied initialization
`imp15_model_day6.npz` was selected upstream using day-6 validation metrics:
`artifacts/imp15_als_trials_day6.json` records nine day-6 iteration/blend
comparisons, and `imp15_als.py` saves the day-6 factor checkpoint conditional
on its validation AUC results. Therefore the V68 day-6 run inherits label-aware
model selection through its initialization. This is an additional reason the
day-6 score is not independent evaluation evidence.

The report's "paired-event 95% interval" is an event-grouped influence-function
normal interval, not a paired bootstrap interval. The local workspace has the
frozen source and report but not the Kaggle model/prediction arrays; the
reported exact replay and AUC metrics cannot be independently recomputed here.
The result is still decisively negative under its reported metrics and is
rejected.

## Frozen recipe and execution

- Factors: 128 dimensions; seed 68128; 6 ALS iterations.
- Initialization: supplied `imp15_model_day6.npz` factors.
- Exposure confidence scale 9; click-decay half-life 3 days.
- Latent regularization 4; bias regularization 20; prior strength 50.
- Fixed rank blend: 95% baseline rank + 5% raw V68 rank.
- Fit only on labeled events strictly before the validation cutoff; warm history
  uses the earliest pre-cutoff event per user and unique user/item/click-time
  triples. No parameter, seed, iteration, or blend search.
- Kaggle reported two Tesla T4 devices. This ALS implementation used four
  Numba CPU threads; the GPU was recorded but not used by the solver.
- Day 6 ran first. Day 5 was skipped because all day-6 gate conditions failed:
  adjusted gain was below +0.0005, the adjusted interval lower bound was
  negative, and ordinary gain was negative.

## Day 6 metrics

| Metric | V68 | V53 baseline | Gain | Paired-event 95% interval for gain |
|---|---:|---:|---:|---:|
| Ordinary AUC | 0.824685203 | 0.825423332 | -0.000738129 | [-0.000897171, -0.000579088] |
| History-adjusted AUC | 0.816445585 | 0.817169046 | -0.000723461 | [-0.000894451, -0.000552471] |

Raw V68 AUC before rank blending was 0.631061178. The run used 588,873
pre-cutoff fitting events and evaluated 35,138 validation events (351,380
impressions). Training took 271.5 seconds.

The saved model replay passed exactly: maximum raw-score error was 0.0, and the
rank-blended predictions were identical on replay. The day-6 checkpoint SHA-256
is `427daad65710fc2055afb0153c17ada2c968cab49e71966ab939b559f0e8367d`.

## Provenance and live-run artifact inventory

Runner SHA-256:
`042d1f57f12c9ada4b3bc82635062c880fb95a2ccd5a6e36759c091c4de9eca8`

Bundle manifest SHA-256:
`61ae1c0035c8667106d0555ea2c2490c99bd33af1f0c8bb47cf588877a8e07a8`

Neural-input manifest SHA-256:
`035233868980ed540d2775a4083a2cac952ba4290b2840b77843099cba45d291`

Initial factor hashes:

- `imp15_model_day5.npz`: `7e78260e0a71f9dcdc304ba8875dd4ed12c911dde6553b2d8aa73d53f9f36db4`
- `imp15_model_day6.npz`: `76fd51e2e2dcf0bc0956eada0844a2c220e651d85266ca6b90255f10c4a5a6d7`

Mounted neural-input hashes:

| File | SHA-256 |
|---|---|
| `events.npz` | `3e2d628a35a9e69c6c75b0a98da28ee9dab96857209b2d16def26ce7d5c8249e` |
| `neural_runner.py` | `270ae4f7d3ee3d674c8b61b6bc17de3c402caff2262a80f989bd28175c1f4c74` |
| `runner.py` | `dabac17aa6eafc30960b003a8d989754f8be6a48671be815179efa14bc491484` |
| `val5.npz` | `6f30b7c3b825a3481006b83f02b81d617b113fb40d4898f34f504b6b9473fd3d` |
| `factors5.npz` | `9a5cc2c6fc1a02da704de10460a5f60225e8af7425c4c3680709437cb9a51a63` |
| `context5.pt` | `2571bc63549e57f123708654cb666a86e62634f0d619dea7bcea1efc6ccb228f` |
| `initial5.npy` | `7571db9be0f599ff1353363546dc765d3a9730ff6040fe5e3d0e90c4b4a28460` |
| `val6.npz` | `963b8f1c8ff1edbeb88955cc087ddeccf8a4784472d488b1270b2eb6deff5d8d` |
| `factors6.npz` | `6b484114d5f57c4444ca4cbde590bf2f506e504652e1bfdb95e6ab6cbe3c087b` |
| `context6.pt` | `fa99607561edb11877a65bd6552a868745fab277867f69698a06486ca5421b73` |
| `initial6.npy` | `aa0a851b75a6c32f6d49e79f68da0a1c18d6c025099118a77120b163f67968a4` |

Kaggle live-run output directory: `/kaggle/working/v68_fixed_als_001`

- `run_config.json` — frozen recipe, device, input hashes, day-6 report, and
  day-5 gate decision.
- `provenance.json` — run and input provenance.
- `outputs_sha256.json` — hashes of the saved run outputs.
- `day6/predictions.npz` — raw scores, rank-blended predictions, baseline,
  labels, weights, event indices, and cold-event flags.
- `day6/report.json` — metrics, intervals, replay status, and checkpoint hash.
- `day6/model.npz` — fitted checkpoint.

These are the files observed in the live run before Quick Save; version 8
contains no committed output files.

Output hashes:

| File | SHA-256 |
|---|---|
| `run_config.json` | `033eb47e319cde14327a0f9406347d37adb6f85d1671df4cebbf1296fdcccb1b` |
| `provenance.json` | `9f6044aeb5396e41259389f9c96a66096b818c6683411999ada556c44bd610d8` |
| `outputs_sha256.json` | `ce8c167dbd9fb660fd40fbbee0af96730c4f78d48c6a2d674663e6b8425731b6` |
| `day6/report.json` | `486e767de48e3994e068a0f29ced505fb877be042e400aa140f993ed7cd4626d` |
| `day6/predictions.npz` | `6840a6050b36a5e67ac0584be57cef958911cc4464ee4b7d6aafa82198596871` |
| `day6/model.npz` | `427daad65710fc2055afb0153c17ada2c968cab49e71966ab939b559f0e8367d` |
