"""Deprecated compatibility import; use :mod:`scd_ml.paths`."""

from scd_ml.paths import *  # noqa: F403

# Historical names retained while old notebooks are migrated.
EDA_ISIC_DIR = RESULTS_DIR / "eda" / "isic"  # noqa: F405
EDA_HAM10000_DIR = RESULTS_DIR / "eda" / "ham10000"  # noqa: F405
MASK_TRAINING_DIR = RESULTS_DIR / "mask_rcnn" / "training"  # noqa: F405
MASK_EVALUATION_DIR = RESULTS_DIR / "mask_rcnn" / "evaluation"  # noqa: F405
MASK_SAMPLES_DIR = RESULTS_DIR / "mask_rcnn" / "samples"  # noqa: F405
