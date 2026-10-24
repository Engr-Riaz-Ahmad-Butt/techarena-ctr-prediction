"""Portable paths. Use a fresh artifact directory when changing datasets."""
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CTR_DATA_DIR", BASE_DIR / "data")).resolve()
TRAIN_CSV = Path(os.environ.get("CTR_TRAIN_CSV", DATA_DIR / "train.csv")).resolve()
EVALUATION_CSV = Path(os.environ.get("CTR_EVALUATION_CSV", DATA_DIR / "evaluation.csv")).resolve()
ARTIFACT_DIR = Path(os.environ.get("CTR_ARTIFACT_DIR", BASE_DIR / "artifacts")).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
