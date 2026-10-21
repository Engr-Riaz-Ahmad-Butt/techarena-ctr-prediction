# TechArena CTR Prediction

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
