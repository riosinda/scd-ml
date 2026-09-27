"""Column contracts for classifier inputs: classes, channel sets and metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

CLASS_ORDER = (
    "Benign-melanocytic",
    "Benign-non-melanocytic",
    "Malignant-melanocytic",
    "Malignant-non-melanocytic",
)
MALIGNANT_CLASSES = ("Malignant-melanocytic", "Malignant-non-melanocytic")

CHANNEL_SETS = {
    "all": ("red", "green", "blue", "gray"),
    "rgb": ("red", "green", "blue"),
    "gray": ("gray",),
}

# ``pixels_x``/``pixels_y`` describe acquisition, not the lesion, and are excluded.
CLINICAL_METADATA = ("age_approx", "sex", "anatom_site_1")

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "image_id",
        "patient_id",
        "lesion_id",
        "group_id",
        "target",
        "split",
        "cv_fold",
        "pixels_x",
        "pixels_y",
        "radiomics_status",
        "radiomics_error",
        "eligible_for_classification",
    }
)


def radiomic_columns(columns: Iterable[str], channel_set: str) -> list[str]:
    """Return radiomic columns whose ``<channel>__`` prefix belongs to ``channel_set``."""
    if channel_set not in CHANNEL_SETS:
        raise ValueError(f"unknown channel set {channel_set!r}; use {sorted(CHANNEL_SETS)}")
    prefixes = tuple(f"{channel}__" for channel in CHANNEL_SETS[channel_set])
    selected = [column for column in columns if column.startswith(prefixes)]
    if not selected:
        raise ValueError(f"no radiomic columns found for channel set {channel_set!r}")
    return selected


def feature_columns(
    columns: Iterable[str], channel_set: str, *, use_metadata: bool
) -> tuple[list[str], list[str]]:
    """Return ``(radiomic, metadata)`` model inputs and refuse identifiers or targets."""
    columns = list(columns)
    radiomics = radiomic_columns(columns, channel_set)
    metadata = list(CLINICAL_METADATA) if use_metadata else []
    missing = sorted(set(metadata) - set(columns))
    if missing:
        raise ValueError(f"cohort is missing clinical metadata columns: {missing}")
    assert_no_forbidden_features([*radiomics, *metadata])
    return radiomics, metadata


def assert_no_forbidden_features(columns: Sequence[str]) -> None:
    leaked = sorted(set(columns) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"identifier or target columns cannot be features: {leaked}")


def encode_target(target: pd.Series) -> np.ndarray:
    """Map class names to integer codes following ``CLASS_ORDER``."""
    unknown = sorted(set(target.dropna().astype(str)) - set(CLASS_ORDER))
    if unknown or target.isna().any():
        raise ValueError(f"target contains missing or unknown classes: {unknown}")
    codes = {name: index for index, name in enumerate(CLASS_ORDER)}
    return target.astype(str).map(codes).to_numpy(dtype=np.int64)
