"""Project-wide path resolver. Portable across Mac, Linux VM, and Windows.

All paths can be overridden with environment variables (useful when data lives
on a mounted disk in the VM, e.g. `SCD_ISIC_DIR=/mnt/disk/isic-archive`).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_or_default(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    return Path(val).expanduser().resolve() if val else default


DATA_DIR    = _env_or_default("SCD_DATA_DIR",    PROJECT_ROOT / "data")
MODELS_DIR  = _env_or_default("SCD_MODELS_DIR",  PROJECT_ROOT / "models")
RESULTS_DIR = _env_or_default("SCD_RESULTS_DIR", PROJECT_ROOT / "results")

HAM10000_DIR = _env_or_default("SCD_HAM10000_DIR", DATA_DIR / "HAM10000")
ISIC_DIR     = _env_or_default("SCD_ISIC_DIR",     DATA_DIR / "isic-archive")

# Generated artifacts live under results/ (data/ is fully gitignored).
PROCESSED_DIR       = RESULTS_DIR / "processed"
EDA_ISIC_DIR        = RESULTS_DIR / "eda" / "isic"
EDA_HAM10000_DIR    = RESULTS_DIR / "eda" / "ham10000"
MASK_TRAINING_DIR   = RESULTS_DIR / "mask_rcnn" / "training"
MASK_EVALUATION_DIR = RESULTS_DIR / "mask_rcnn" / "evaluation"
MASK_SAMPLES_DIR    = RESULTS_DIR / "mask_rcnn" / "samples"
SEGMENTATION_DIR    = RESULTS_DIR / "segmentation"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
