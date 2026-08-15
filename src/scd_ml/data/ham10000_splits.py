"""Group-safe HAM10000 split used only by the segmentation task."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .schemas import assert_disjoint_groups, require_columns, require_unique


def build_ham10000_manifest(metadata: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Create approximate 70/10/20 row proportions while keeping lesions intact."""
    require_columns(metadata, ["image_id", "lesion_id"], name="HAM10000 metadata")
    frame = metadata[["image_id", "lesion_id"]].copy()
    require_unique(frame, "image_id", name="HAM10000 metadata")
    if frame[["image_id", "lesion_id"]].isna().any().any():
        raise ValueError("HAM10000 image_id and lesion_id cannot be null")

    frame["image_id"] = frame["image_id"].astype(str)
    frame["lesion_id"] = frame["lesion_id"].astype(str)
    frame["group_id"] = "lesion:" + frame["lesion_id"]

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    development_pos, test_pos = next(outer.split(frame, groups=frame["group_id"]))
    development = frame.iloc[development_pos]

    # 12.5% of the remaining 80% gives an overall validation share of 10%.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_relative, val_relative = next(
        inner.split(development, groups=development["group_id"])
    )

    frame["split"] = "test"
    frame.iloc[development_pos[train_relative], frame.columns.get_loc("split")] = "train"
    frame.iloc[development_pos[val_relative], frame.columns.get_loc("split")] = "val"
    frame["image_filename"] = frame["image_id"] + ".jpg"
    frame["mask_filename"] = frame["image_id"] + "_segmentation.png"

    manifest = frame[
        [
            "image_id",
            "lesion_id",
            "group_id",
            "split",
            "image_filename",
            "mask_filename",
        ]
    ].sort_values("image_id").reset_index(drop=True)
    assert_disjoint_groups(manifest)
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise RuntimeError("HAM10000 split must contain train, val, and test")
    return manifest
