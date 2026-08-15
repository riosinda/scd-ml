"""Leakage-safe ISIC train/test and development-fold assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .schemas import assert_disjoint_groups, require_columns, require_unique


def _present(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def make_group_id(row: pd.Series) -> str:
    """Build the patient → lesion → image fallback group required by the protocol."""
    if _present(row.get("patient_id")):
        return f"patient:{row['patient_id']}"
    if _present(row.get("lesion_id")):
        return f"lesion:{row['lesion_id']}"
    return f"image:{row['image_id']}"


def _best_test_indices(frame: pd.DataFrame, seed: int, n_splits: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_distribution = frame["target"].value_counts(normalize=True)
    target_fraction = 1.0 / n_splits
    candidates: list[tuple[float, np.ndarray]] = []

    try:
        folds = splitter.split(frame, y=frame["target"], groups=frame["group_id"])
        for _, test_idx in folds:
            test = frame.iloc[test_idx]
            size_error = abs(len(test) / len(frame) - target_fraction)
            fold_distribution = test["target"].value_counts(normalize=True)
            class_error = (
                global_distribution.sub(fold_distribution, fill_value=0).abs().sum()
            )
            candidates.append((size_error + class_error, test_idx))
    except ValueError as exc:
        raise ValueError(
            "ISIC cannot be split into five stratified group folds. Each class needs "
            "enough independent patient/lesion groups."
        ) from exc

    return min(candidates, key=lambda item: item[0])[1]


def build_isic_manifest(
    metadata: pd.DataFrame,
    *,
    image_id_column: str = "isic_id",
    target_column: str = "target",
    seed: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Return an ISIC 80/20 manifest without imputing or transforming features."""
    require_columns(metadata, [image_id_column], name="ISIC metadata")
    if n_splits != 5:
        raise ValueError("The locked ISIC protocol requires exactly five folds")

    selected = metadata.copy()
    if target_column not in selected:
        require_columns(
            selected,
            ["diagnosis_1", "melanocytic"],
            name="ISIC metadata without a precomputed target",
        )
        diagnosis = selected["diagnosis_1"].astype("string").str.strip().str.title()
        lineage_raw = selected["melanocytic"]
        lineage = lineage_raw.map(
            {
                True: "melanocytic",
                False: "non-melanocytic",
                1: "melanocytic",
                0: "non-melanocytic",
                "true": "melanocytic",
                "false": "non-melanocytic",
                "True": "melanocytic",
                "False": "non-melanocytic",
            }
        )
        valid = diagnosis.isin(["Benign", "Malignant"]) & lineage.notna()
        selected = selected.loc[valid].copy()
        selected[target_column] = diagnosis.loc[valid] + "-" + lineage.loc[valid]

    rename = {image_id_column: "image_id", target_column: "target"}
    selected = selected.rename(columns=rename)
    for optional in ("patient_id", "lesion_id"):
        if optional not in selected:
            selected[optional] = pd.NA

    require_unique(selected, "image_id", name="ISIC metadata")
    if selected["image_id"].isna().any():
        raise ValueError("ISIC image IDs cannot be null")
    if selected["target"].isna().any():
        raise ValueError("ISIC target cannot be null when constructing a stratified split")

    selected["image_id"] = selected["image_id"].astype(str)
    selected["group_id"] = selected.apply(make_group_id, axis=1)
    selected["split"] = "train"
    selected["cv_fold"] = pd.Series(pd.NA, index=selected.index, dtype="Int64")

    test_positions = _best_test_indices(selected, seed, n_splits)
    selected.iloc[test_positions, selected.columns.get_loc("split")] = "test"

    development = selected.loc[selected["split"] == "train"]
    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        for fold, (_, val_positions) in enumerate(
            inner.split(development, y=development["target"], groups=development["group_id"])
        ):
            indices = development.iloc[val_positions].index
            selected.loc[indices, "cv_fold"] = fold
    except ValueError as exc:
        raise ValueError(
            "The ISIC development set cannot support five stratified group folds"
        ) from exc

    output_columns = [
        "image_id",
        "patient_id",
        "lesion_id",
        "group_id",
        "target",
        "split",
        "cv_fold",
    ]
    manifest = selected[output_columns].sort_values("image_id").reset_index(drop=True)
    assert_disjoint_groups(manifest)
    if manifest.loc[manifest["split"] == "train", "cv_fold"].isna().any():
        raise RuntimeError("Every ISIC training row must have a development fold")
    if manifest.loc[manifest["split"] == "test", "cv_fold"].notna().any():
        raise RuntimeError("ISIC test rows must not have a development fold")
    return manifest
