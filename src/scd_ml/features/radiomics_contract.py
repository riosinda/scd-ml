"""Validate the CSV hand-off from the isolated Python 3.7 extractor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scd_ml.data.schemas import require_columns, require_unique


def validate_radiomics_contract(
    masks_manifest: str | Path,
    features_csv: str | Path,
    status_csv: str | Path,
) -> dict[str, int]:
    masks = pd.read_csv(masks_manifest, dtype={"image_id": str})
    features = pd.read_csv(features_csv, dtype={"image_id": str})
    statuses = pd.read_csv(status_csv, dtype={"image_id": str})
    require_columns(masks, ["image_id", "status"], name="mask manifest")
    require_columns(features, ["image_id"], name="radiomics features")
    require_columns(statuses, ["image_id", "status", "error"], name="radiomics status")
    require_unique(masks, "image_id", name="mask manifest")
    require_unique(features, "image_id", name="radiomics features")
    require_unique(statuses, "image_id", name="radiomics status")

    valid_status = statuses["status"].isin(["ok", "empty_mask", "error"]) | statuses[
        "status"
    ].str.startswith("upstream_", na=False)
    if not valid_status.all():
        invalid = sorted(statuses.loc[~valid_status, "status"].astype(str).unique())
        raise ValueError(f"Unknown radiomics statuses: {invalid}")

    expected = set(masks["image_id"])
    observed = set(statuses["image_id"])
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            f"Radiomics status does not cover the mask manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )

    successful = set(statuses.loc[statuses["status"] == "ok", "image_id"])
    feature_ids = set(features["image_id"])
    if feature_ids != successful:
        missing = sorted(successful - feature_ids)[:5]
        unexpected = sorted(feature_ids - successful)[:5]
        raise ValueError(
            f"Feature rows and successful statuses differ; missing={missing}, "
            f"unexpected={unexpected}"
        )
    feature_columns = [column for column in features.columns if column != "image_id"]
    if successful and not feature_columns:
        raise ValueError("Successful extractions must contain radiomic feature columns")
    for column in feature_columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Radiomics feature {column!r} contains non-numeric values")

    return {
        "expected_images": len(expected),
        "successful_images": len(successful),
        "failed_images": len(expected - successful),
        "feature_columns": len(feature_columns),
    }
