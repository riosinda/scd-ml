"""Load the validated classification cohort from the locked pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scd_ml.features.radiomics_contract import validate_radiomics_contract
from scd_ml.paths import FEATURES_DIR, PROCESSED_DIR, SEGMENTATION_DIR, SPLITS_DIR

from .datasets import build_classification_cohort


@dataclass(frozen=True)
class CohortInputs:
    manifest: Path = SPLITS_DIR / "isic_classification.csv"
    masks_manifest: Path = SEGMENTATION_DIR / "isic_masks_manifest.csv"
    features: Path = FEATURES_DIR / "radiomics_features.csv"
    status: Path = FEATURES_DIR / "radiomics_status.csv"
    metadata: Path = PROCESSED_DIR / "isic_model_input_raw.parquet"


def load_cohort(inputs: CohortInputs | None = None) -> pd.DataFrame:
    """Validate the radiomics hand-off and return one row per manifest image.

    Rows whose extraction failed stay in the cohort with
    ``eligible_for_classification=False`` so callers can report coverage.
    """
    inputs = inputs or CohortInputs()
    missing = [str(path) for path in vars(inputs).values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing classification inputs: {missing}")
    validate_radiomics_contract(inputs.masks_manifest, inputs.features, inputs.status)
    return build_classification_cohort(
        pd.read_csv(inputs.manifest, dtype={"image_id": "string"}),
        pd.read_csv(inputs.features, dtype={"image_id": "string"}),
        pd.read_csv(inputs.status, dtype={"image_id": "string"}),
        metadata=pd.read_parquet(inputs.metadata),
    )


def coverage_by_class(cohort: pd.DataFrame) -> pd.DataFrame:
    """Total, eligible and excluded images per split and class."""
    coverage = (
        cohort.groupby(["split", "target"], observed=True)
        .agg(
            total_images=("image_id", "size"),
            eligible_images=("eligible_for_classification", "sum"),
        )
        .reset_index()
    )
    coverage["eligible_images"] = coverage["eligible_images"].astype(int)
    coverage["excluded_images"] = coverage["total_images"] - coverage["eligible_images"]
    coverage["exclusion_pct"] = (
        100 * coverage["excluded_images"] / coverage["total_images"]
    ).round(2)
    return coverage
