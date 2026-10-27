# TechArena CTR Prediction

Use this single project folder for all work. Source code and Git history are here; prepared data, model weights and runtime reports are under `artifacts/`. Submission CSVs are local, excluded from Git. The former sibling `ctr_work` folder has been consolidated into this project.

Current audited local candidate: **V53**, history-adjusted AUC **0.817169** on the January 26 cohort; ordinary AUC **0.825423**. Its day5 check uses user cross-fitting, followed by a genuinely later day6 check. These dates were repeatedly reused and are predominantly weekends, unlike evaluation. See [latest candidate](results/latest_local_candidate.json) for the current version and [official score ledger](results/official_submission_scores.json) for competition results. Local ordinary AUC, adjusted AUC and official AUC are different quantities.

Last user-confirmed **official score: V35 = 0.816623**. The user confirms V36 and V39 have not been uploaded. **V39** is the latest exported and verified CSV (`submission_v39.csv` in the local project root). V53 has not been exported. The requested score **0.834786** remains unmet; another upload is not recommended before local adjusted AUC reaches the user's minimum **0.82**. This threshold does not guarantee an official score.

## Historical V20 baseline notes

At V20, the tree learner with aligned exposure-ALS interactions achieved ordinary local AUC **0.821528 / 0.814600** on January 26 / January 25. Frozen settings were 735 trees and a 50% rank blend with the preceding ALS-feature ensemble. Independent feature, saved-model and full-recipe checks passed (`results/imp19_test_candidate_report.json`). These are historical metrics; current candidate and official results are listed above.

The Bernoulli/NCE experiment used this independently accepted baseline, frozen by file hashes before fitting (`results/imp18_baseline.json`). Its gradients, duplicate updates, production float32 arithmetic, extreme logits and parameter isolation passed independent tests. All nine logged epoch/blend settings failed the improvement gate, so no confirmation fit or promotion occurred (`results/imp18_test_rejection_report.json`). Rejected raw predictions were not retained; trial metrics are logged results rather than independently replayed scores.

The accepted experiment appends 32 aligned ALS component interactions (`imp19_plan.md`). Independent feature replay passed on 1,000 rows, including cold users and empty history; the first 322 features were unchanged. Rotation, strict-past histories and temporal guard checks passed, including a separate run of the portable source. Both validation dates passed with positive conditional improvement intervals.

`imp19_finalize.py` refits the exact accepted dependency chain on final rolling days4/5/6 and creates `submission_v20_candidate.csv`. Frozen source/input hashes protect resumable stage outputs. `imp19_test_final.py --evaluation-csv data/evaluation.csv` independently checks original row IDs, the complete blend, model replays and file hashes before promotion to `submission_v20.csv`. The final refit, independent audit and exclusive promotion to `submission_v20.csv` are complete; see `results/imp19_test_final_report.json` and `results/imp19_v20_promotion_report.json`. Its official score is unknown. The output is529390rows (30317514bytes), SHA256 `8f3bb5e49f96e49efc4d6416e9c593f27f118aef06c14e19b85ead45291c789b`. The CSV itself stays outside Git.

A subsequent capacity experiment is prepared (`imp20_capacity_plan.md`): only `num_leaves` changes from31 to63 on the same354 features. Independent and portable isolated tests verify this difference and reject changes to either baseline member, the accepted audit or the confirmation manifest. It has not been fitted; the requested V20 export has now been delivered before any capacity fit.

Recent experiments preserve the same first-event validation population. The contextual warm-sequence model, fixed-item recency adaptation, and first-event-only training failed to improve the current baseline. See the measured reports in `results/imp7_sequence_report.json`, `results/imp8_foldin_report.json`, and `results/imp9_firstevents_report.json`. LightGBM passed both dates with 641 trees and a frozen 50% probability blend. Independent saved-model replay on 1,000 rows per date and full prediction/blend/metric checks passed; see `results/imp10_test_candidate_report.json`. Dense latent calibration then added a frozen 10% blend at epoch 6; full saved-model, causal feature, blend and metric replay passed on both dates (`results/imp11_test_candidate_report.json`). The later context-dependent calibration and neural learner failed the two-date gate; a constant-input neural mask produced identical predictions. Neither replaces the accepted baseline. See `results/imp12_context_report.json`, `results/imp13_test_candidate_report.json` and `results/imp14_identity_diagnostic.json`. Exposure-aware ALS with a fixed 5% rank blend improved both dates; saved-model, cold-user fold-in and full rank-recipe audits passed (`results/imp15_test_candidate_report.json`). Repeated validation use still limits score certainty.

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

Run `python choice_experiment.py` after the preceding required stages. Local January 26 / January 25 ROC AUC: **0.805178 / 0.794261**. Blend 25% supervised event-choice model with 75% V6; four training epochs. Historical V8 artifact: artifacts/submission_v8.csv.

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
