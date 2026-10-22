# TechArena CTR Prediction

Latest verified local model: **V18**, AUC **0.816800** on January 26 and **0.808297** on January 25. The requested 0.82 threshold has not been reached. Its official competition score is unknown; V17 was not uploaded. See [V18 decomposition](docs/v18-decomposition.md) for validation and reproduction. Datasets, model weights and submission CSVs are excluded from Git.

Click-through probability prediction for the TechArena challenge. Each event contains ten candidates and at most one click. Models combine user histories, collaborative features and CatBoost. Author: Riaz Ahmad Butt.

## Setup

Python 3.12; CPU only. Create a virtual environment and install `requirements.txt`. Place the competition files in `data/train.csv` and `data/evaluation.csv`. Obtain the dataset separately from the organizers.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare.py
python model.py
```

Generated arrays, models and predictions go into `artifacts/` and are excluded from Git. Override locations with `CTR_TRAIN_CSV`, `CTR_EVALUATION_CSV` and `CTR_ARTIFACT_DIR` environment variables. Use a fresh artifact directory if inputs change: caches are keyed by time cutoff, not dataset hash.

The scripts are competition experiments and assume the original dataset sizes and contiguous ten-candidate events. Importing the shared core modules loads prepared arrays; run preparation first. Full experiment reproduction takes substantially longer than a single baseline run.

## Baseline

V1 uses smoothed click rates and history similarities from sparse SVD embeddings. Local January 26 ROC AUC: **0.702969**. Higher AUC is better. Validation is each user's first event on the selected date; features and classifier fitting use preceding dates. Early stopping uses this validation window, so this is not an untouched final test.

Historical metrics are recorded in `results/`. Runtime metrics are written to `artifacts/`. Dataset files and trained models are not distributed.

## V2

Run `python improve.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.731177 / 0.716325**. 64-dimensional factors, recency and candidate-relative features. User-reported official public score: 0.741028.

## V3

Run `python prepare_times.py` then `python timing_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.731246 / 0.716220**. Rejected: improvement did not transfer to both dates. This stage also produces base statistics caches used downstream.

## V4

Run `python relations_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.742770 / 0.724951**. Directed click-pair counts, lift and latent relationships.

## V5

Run `python warm_context_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.758172 / 0.747077**. Warm-up history pairs and smoothed user/item contextual rates.

## V6

Run `python preference_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.800327 / 0.786659**. Richer item factors and exposure-adjusted user profiles.

## MF experiment (optional)

Run `python factorization_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.800555 / 0.786745**. Factorization features failed to beat the matched control on both dates. This optional branch is not needed for V8.

## V7

Run `python ridge_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.801756 / 0.788564**. Blend 25% ridge-feature model with 75% V6.

## V8

Run `python choice_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.805178 / 0.794261**. Blend 25% supervised event-choice model with 75% V6; four training epochs. Current selected submission: artifacts/submission_v8.csv.

## Reproduction and submission checks

Run the stages above in order; the MF experiment is optional. Run V7 before `select_ensemble.py`; V8 itself depends on V6, not V7. The V7/V8 combination was rejected because it reduced AUC on January 25. Repeated use of these overlapping temporal windows can overfit model selection. The latest local scores are not official leaderboard scores, and 0.82 has not been achieved in the recorded results.

```powershell
python select_ensemble.py
python check_submission.py artifacts/submission_v8.csv
```

The checker verifies the header, all evaluation IDs, duplicates, finite probabilities, range and expected row count, and writes a checksum report beside the CSV. Run Python normally, without `-O`, because validation uses assertions. Upload the resulting CSV through the competition website; these scripts do not upload automatically.

## Verification and provenance

The published source has portable path changes relative to the completed local experiments. All Python files passed syntax compilation, and the portable checker validated the existing V8 output: 529,390 rows, zero missing IDs, duplicates or invalid probabilities. Full training was not repeated for this repository packaging step. Dependency pins record the local experiment versions; a clean dependency installation has not been retested.

Commits organize previously completed experiments into separate reviewable stages; their timestamps reflect repository preparation. Historical scores are retained in results JSON files, including rejected experiments. Do not load pickle model caches from untrusted sources.

## Additional experiment: stacked-day training

The optional stack_experiment.py trains on multiple chronological blocks with statistics computed before each block. It requires the V6 pipeline. Completed results are now recorded in results/validation_stack.json. The standalone model failed the two-date gate, but proved useful in a blend.


## Latest selected candidate

Run `stack_final.py` after the stacked validation experiment, then `blend_select.py`, `competition_experiment.py` and `nonlinear_experiment.py` in order. The latest selected file is **artifacts/submission_nonlinear.csv**. These stages require the V6, V7 and V8 outputs described above.

| Candidate | January 26 AUC | January 25 AUC |
|---|---:|---:|
| Three-way blend | 0.809385 | 0.795750 |
| Competition adjustment | 0.809829 | 0.796671 |
| Log-odds ensemble | 0.810105 | 0.797186 |

The log-odds ensemble weights are stack 0.45, V6 0.27, choice 0.18 and V7 0.10, followed by a 50% event-normalized-odds adjustment. Settings were selected from 37 configurations on January 26 and frozen for January 25. Approximate paired event-cluster gain intervals are recorded, but do not account for repeated model selection on these validation dates. These are local results; the official leaderboard score is unknown. The 529,390-row output passed the independent submission checker. Previous files are preserved.

A further 64-dimensional choice model was trained for six passes at learning rate 0.03 and regularization 0.02. Blend weights 0.05, 0.10, 0.20 and 0.30 failed to improve the January 26 score, so it was rejected before confirmation-date testing and final fitting. Source: choice64_experiment.py; result: results/validation_choice64.json. The log-odds ensemble remains selected.

## V11 experimental candidate

`weighted_experiment.py` compares a bounded grid of log-odds ensemble weights on January 26, then applies the selected weights unchanged on January 25. Weights (stack, V6, choice, V7): 0.45, 0.15, 0.20, 0.20. Output: **artifacts/submission_v11.csv**. V10 is an identical versioned copy of submission_nonlinear.csv; preserve it for comparison.

| Validation | V10 | V11 |
|---|---:|---:|
| January 26 | 0.810104863 | 0.810155847 |
| January 25 | 0.797186395 | 0.797285247 |

This is a very small gain. The approximate paired January 26 improvement interval includes zero; repeated selection on reused windows adds further uncertainty. V11 is an experimental alternative, not a proven leaderboard improvement or a reason to discard V10. The event sharpness/propensity adjustment in event_propensity_experiment.py was also tested and rejected because it did not beat V10. Results for both experiments are recorded separately.
