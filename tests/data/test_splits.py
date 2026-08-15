from __future__ import annotations

import unittest

import pandas as pd

from scd_ml.data.ham10000_splits import build_ham10000_manifest
from scd_ml.data.isic_splits import build_isic_manifest


class IsicSplitTests(unittest.TestCase):
    @staticmethod
    def metadata() -> pd.DataFrame:
        rows = []
        for class_index in range(4):
            for patient_index in range(20):
                for image_index in range(2):
                    rows.append(
                        {
                            "isic_id": f"i-{class_index}-{patient_index}-{image_index}",
                            "patient_id": f"p-{class_index}-{patient_index}",
                            "lesion_id": f"l-{class_index}-{patient_index}",
                            "target": f"class-{class_index}",
                            "age_approx": None if image_index == 0 else 50,
                            "sex": None if image_index == 0 else "female",
                        }
                    )
        return pd.DataFrame(rows)

    def test_isic_is_group_safe_and_keeps_test_near_twenty_percent(self) -> None:
        metadata = self.metadata()
        manifest = build_isic_manifest(metadata)
        fraction = (manifest["split"] == "test").mean()
        self.assertAlmostEqual(fraction, 0.20, delta=0.03)
        memberships = manifest.groupby("group_id")["split"].nunique()
        self.assertEqual(memberships.max(), 1)
        self.assertTrue(manifest.loc[manifest["split"] == "test", "cv_fold"].isna().all())
        self.assertTrue(manifest.loc[manifest["split"] == "train", "cv_fold"].notna().all())
        self.assertTrue(metadata["age_approx"].isna().any())
        self.assertTrue(metadata["sex"].isna().any())

    def test_group_fallback_order(self) -> None:
        metadata = self.metadata()
        metadata.loc[0, "patient_id"] = None
        metadata.loc[0, "lesion_id"] = "fallback-lesion"
        metadata.loc[1, ["patient_id", "lesion_id"]] = None
        manifest = build_isic_manifest(metadata)
        groups = manifest.set_index("image_id")["group_id"]
        self.assertEqual(groups.loc[metadata.loc[0, "isic_id"]], "lesion:fallback-lesion")
        second_image_id = metadata.loc[1, "isic_id"]
        self.assertEqual(groups.loc[second_image_id], f"image:{second_image_id}")

    def test_target_can_be_derived_without_clinical_imputation(self) -> None:
        metadata = self.metadata().drop(columns="target")
        class_number = metadata["isic_id"].str.split("-").str[1].astype(int)
        metadata["diagnosis_1"] = class_number.map(
            {0: "Benign", 1: "Benign", 2: "Malignant", 3: "Malignant"}
        )
        metadata["melanocytic"] = class_number.isin([0, 2])
        manifest = build_isic_manifest(metadata)
        self.assertEqual(
            set(manifest["target"]),
            {
                "Benign-melanocytic",
                "Benign-non-melanocytic",
                "Malignant-melanocytic",
                "Malignant-non-melanocytic",
            },
        )


class HamSplitTests(unittest.TestCase):
    def test_ham_is_group_safe_and_drops_diagnosis(self) -> None:
        rows = []
        for lesion_index in range(100):
            for image_index in range(1 + (lesion_index % 3 == 0)):
                rows.append(
                    {
                        "image_id": f"ham-{lesion_index}-{image_index}",
                        "lesion_id": f"lesion-{lesion_index}",
                        "dx": "must-not-be-used",
                    }
                )
        manifest = build_ham10000_manifest(pd.DataFrame(rows))
        self.assertNotIn("dx", manifest.columns)
        self.assertEqual(manifest.groupby("group_id")["split"].nunique().max(), 1)
        fractions = manifest["split"].value_counts(normalize=True)
        self.assertAlmostEqual(fractions["train"], 0.70, delta=0.06)
        self.assertAlmostEqual(fractions["val"], 0.10, delta=0.04)
        self.assertAlmostEqual(fractions["test"], 0.20, delta=0.05)


if __name__ == "__main__":
    unittest.main()
