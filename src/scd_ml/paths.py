"""Project paths with optional ``SCD_*`` environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_or_default(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_or_default("SCD_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _env_or_default("SCD_MODELS_DIR", PROJECT_ROOT / "models")
RESULTS_DIR = _env_or_default("SCD_RESULTS_DIR", PROJECT_ROOT / "results")
HAM10000_DIR = _env_or_default("SCD_HAM10000_DIR", DATA_DIR / "HAM10000")
ISIC_DIR = _env_or_default("SCD_ISIC_DIR", Path("~/data/gcs").expanduser())
ISIC_IMAGES_DIR = _env_or_default("SCD_ISIC_IMAGES_DIR", ISIC_DIR)
ISIC_MASKS_DIR = _env_or_default(
    "SCD_ISIC_MASKS_DIR", Path("~/data/gcs-masks").expanduser()
)

SPLITS_DIR = RESULTS_DIR / "splits"
PROCESSED_DIR = RESULTS_DIR / "processed"
SEGMENTATION_DIR = RESULTS_DIR / "segmentation"
SEGMENTATION_TRAINING_DIR = SEGMENTATION_DIR / "training"
SEGMENTATION_EVALUATION_DIR = SEGMENTATION_DIR / "evaluation"
FEATURES_DIR = RESULTS_DIR / "features"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
MLFLOW_DIR = RESULTS_DIR / "mlflow"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
