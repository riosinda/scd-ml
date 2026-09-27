"""Small synthetic classification cohort shared by the classification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scd_ml.classification.columns import CLASS_ORDER

FEATURES = ("firstorder_Mean", "firstorder_Energy", "glcm_Contrast", "glszm_ZoneEntropy")


def synthetic_cohort(n_per_fold: int = 40, n_test: int = 40, seed: int = 0) -> pd.DataFrame:
    """Cohort with class signal, missing metadata, one failed extraction and 5 folds."""
    rng = np.random.default_rng(seed)
    n_development = 5 * n_per_fold
    n = n_development + n_test
    target_codes = np.arange(n) % len(CLASS_ORDER)
    cohort = pd.DataFrame(
        {
            "image_id": [f"image-{index:04d}" for index in range(n)],
            "group_id": [f"patient:{index}" for index in range(n)],
            "target": np.asarray(CLASS_ORDER)[target_codes],
            "split": ["train"] * n_development + ["test"] * n_test,
            "cv_fold": pd.array(
                [index % 5 for index in range(n_development)] + [pd.NA] * n_test,
                dtype="Int64",
            ),
            "age_approx": rng.choice([35.0, 50.0, 65.0, np.nan], size=n),
            "sex": rng.choice(["male", "female", None], size=n),
            "anatom_site_1": rng.choice(["Trunk", "Head and neck", None], size=n),
            "pixels_x": 1024,
            "pixels_y": 768,
        }
    )
    for channel in ("red", "green", "blue", "gray"):
        for position, feature in enumerate(FEATURES):
            signal = target_codes * (position + 1)
            cohort[f"{channel}__original_{feature}"] = signal + rng.normal(size=n)
    status = np.where(np.arange(n) == 3, "empty_mask", "ok")
    cohort["radiomics_status"] = status
    cohort["eligible_for_classification"] = status == "ok"
    radiomic = [column for column in cohort.columns if "__original_" in column]
    cohort.loc[~cohort["eligible_for_classification"], radiomic] = np.nan
    return cohort
