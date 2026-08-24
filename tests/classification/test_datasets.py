from __future__ import annotations

import unittest

import pandas as pd

from scd_ml.classification.datasets import (
    build_classification_cohort,
    split_development_fold,
)


class ClassificationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.DataFrame(
            {
                "image_id": [f"image-{index}" for index in range(6)],
                "patient_id": [f"patient-{index}" for index in range(6)],
                "lesion_id": [f"lesion-{index}" for index in range(6)],
                "group_id": [f"patient:patient-{index}" for index in range(6)],
                "target": ["a", "b", "a", "b", "a", "b"],
                "split": ["train"] * 5 + ["test"],
                "cv_fold": pd.Series([0, 1, 2, 3, 4, pd.NA], dtype="Int64"),
            }
        )
        self.statuses = pd.DataFrame(
            {
                "image_id": self.manifest["image_id"],
                "status": ["ok", "ok", "empty_mask", "ok", "ok", "ok"],
                "error": [""] * 6,
            }
        )
        successful = self.statuses.loc[self.statuses["status"].eq("ok"), "image_id"]
        self.features = pd.DataFrame(
            {
                "image_id": successful,
                "gray__original_firstorder_Mean": range(len(successful)),
            }
        )

    def test_failed_extractions_remain_visible_and_are_not_eligible(self) -> None:
        cohort = build_classification_cohort(
            self.manifest, self.features, self.statuses
        )

        self.assertEqual(len(cohort), len(self.manifest))
        failed = cohort.loc[cohort["image_id"].eq("image-2")].iloc[0]
        self.assertEqual(failed["radiomics_status"], "empty_mask")
        self.assertFalse(failed["eligible_for_classification"])

        training, validation = split_development_fold(cohort, 0)
        self.assertEqual(set(validation["image_id"]), {"image-0"})
        self.assertNotIn("image-2", set(training["image_id"]))

    def test_partial_status_coverage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover"):
            build_classification_cohort(
                self.manifest, self.features, self.statuses.iloc[:-1]
            )

    def test_raw_metadata_is_preserved_including_missing_values(self) -> None:
        metadata = pd.DataFrame(
            {
                "isic_id": self.manifest["image_id"],
                "age_approx": [40, pd.NA, 55, 60, 35, 70],
                "anatom_site_1": ["Trunk", pd.NA, "Head", "Arm", "Leg", "Trunk"],
                "pixels_x": [1024] * 6,
                "pixels_y": [768] * 6,
                "sex": ["female", pd.NA, "male", "female", "male", "female"],
            }
        )

        cohort = build_classification_cohort(
            self.manifest,
            self.features,
            self.statuses,
            metadata=metadata,
        )

        observed = cohort.set_index("image_id")
        self.assertEqual(observed.loc["image-0", "age_approx"], 40)
        self.assertTrue(pd.isna(observed.loc["image-1", "age_approx"]))
        self.assertTrue(pd.isna(observed.loc["image-1", "anatom_site_1"]))
        self.assertTrue(pd.isna(observed.loc["image-1", "sex"]))

    def test_missing_requested_metadata_column_is_rejected(self) -> None:
        incomplete_metadata = pd.DataFrame(
            {
                "isic_id": self.manifest["image_id"],
                "age_approx": [40] * 6,
            }
        )

        with self.assertRaisesRegex(ValueError, "ISIC model metadata"):
            build_classification_cohort(
                self.manifest,
                self.features,
                self.statuses,
                metadata=incomplete_metadata,
            )


if __name__ == "__main__":
    unittest.main()
