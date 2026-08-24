"""Build the auditable classification cohort and development-fold views."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from scd_ml.data.schemas import assert_disjoint_groups, require_columns, require_unique


DEFAULT_METADATA_COLUMNS = (
    "age_approx",
    "anatom_site_1",
    "pixels_x",
    "pixels_y",
    "sex",
)


def _string_ids(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["image_id"] = output["image_id"].astype("string")
    return output


def build_classification_cohort(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    statuses: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    metadata_id_column: str = "isic_id",
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Return one row per manifest image with explicit radiomics eligibility.

    This function refuses partial status coverage. Failed radiomics rows remain in
    the cohort with ``eligible_for_classification=False`` instead of disappearing
    through an inner join.
    """
    require_columns(
        manifest,
        ["image_id", "group_id", "target", "split", "cv_fold"],
        name="ISIC classification manifest",
    )
    require_columns(features, ["image_id"], name="radiomics features")
    require_columns(statuses, ["image_id", "status", "error"], name="radiomics status")
    require_unique(manifest, "image_id", name="ISIC classification manifest")
    require_unique(features, "image_id", name="radiomics features")
    require_unique(statuses, "image_id", name="radiomics status")

    manifest = _string_ids(manifest)
    features = _string_ids(features)
    statuses = _string_ids(statuses)

    expected_ids = set(manifest["image_id"])
    status_ids = set(statuses["image_id"])
    if status_ids != expected_ids:
        missing = sorted(expected_ids - status_ids)[:5]
        unexpected = sorted(status_ids - expected_ids)[:5]
        raise ValueError(
            "radiomics status does not cover the classification manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )

    successful_ids = set(statuses.loc[statuses["status"] == "ok", "image_id"])
    feature_ids = set(features["image_id"])
    if feature_ids != successful_ids:
        missing = sorted(successful_ids - feature_ids)[:5]
        unexpected = sorted(feature_ids - successful_ids)[:5]
        raise ValueError(
            "radiomics features and successful statuses differ; "
            f"missing={missing}, unexpected={unexpected}"
        )

    valid_splits = set(manifest["split"].dropna().astype(str))
    if not valid_splits <= {"train", "test"}:
        raise ValueError(f"classification manifest has invalid splits: {sorted(valid_splits)}")
    development = manifest["split"].eq("train")
    if manifest.loc[development, "cv_fold"].isna().any():
        raise ValueError("every development row must have a cv_fold")
    if manifest.loc[~development, "cv_fold"].notna().any():
        raise ValueError("test rows must not have a cv_fold")
    observed_folds = set(manifest.loc[development, "cv_fold"].astype(int))
    if observed_folds != set(range(5)):
        raise ValueError(f"classification manifest must contain folds 0-4, got {observed_folds}")
    assert_disjoint_groups(manifest)

    renamed_statuses = statuses[["image_id", "status", "error"]].rename(
        columns={"status": "radiomics_status", "error": "radiomics_error"}
    )
    cohort = manifest.merge(renamed_statuses, on="image_id", how="left", validate="one_to_one")

    if metadata is not None:
        require_columns(metadata, [metadata_id_column], name="ISIC model metadata")
        requested_metadata = list(metadata_columns)
        require_columns(
            metadata,
            requested_metadata,
            name="ISIC model metadata",
        )
        metadata = metadata.rename(columns={metadata_id_column: "image_id"})
        metadata = _string_ids(metadata)
        require_unique(metadata, "image_id", name="ISIC model metadata")
        cohort = cohort.merge(
            metadata[["image_id", *requested_metadata]],
            on="image_id",
            how="left",
            validate="one_to_one",
        )

    cohort = cohort.merge(features, on="image_id", how="left", validate="one_to_one")
    cohort["eligible_for_classification"] = cohort["radiomics_status"].eq("ok")
    cohort["cv_fold"] = cohort["cv_fold"].astype("Int64")
    return cohort.sort_values("image_id").reset_index(drop=True)


def split_development_fold(
    cohort: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return eligible train/validation rows for one locked development fold."""
    if fold not in range(5):
        raise ValueError("fold must be one of 0, 1, 2, 3 or 4")
    require_columns(
        cohort,
        ["split", "cv_fold", "group_id", "eligible_for_classification"],
        name="classification cohort",
    )
    eligible_development = cohort[
        cohort["eligible_for_classification"] & cohort["split"].eq("train")
    ]
    validation = eligible_development[eligible_development["cv_fold"].eq(fold)].copy()
    training = eligible_development[~eligible_development["cv_fold"].eq(fold)].copy()
    overlap = set(training["group_id"]) & set(validation["group_id"])
    if overlap:
        raise RuntimeError(f"group leakage in fold {fold}: {sorted(overlap)[:5]}")
    if training.empty or validation.empty:
        raise ValueError(f"fold {fold} produced an empty train or validation partition")
    return training.reset_index(drop=True), validation.reset_index(drop=True)
