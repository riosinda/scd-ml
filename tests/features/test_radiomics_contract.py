from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scd_ml.features.radiomics_contract import validate_radiomics_contract


class RadiomicsContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.masks = self.root / "masks.csv"
        self.features = self.root / "features.csv"
        self.status = self.root / "status.csv"
        pd.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "status": ["segmented", "no_detection", "error"],
            }
        ).to_csv(self.masks, index=False)
        pd.DataFrame({"image_id": ["a"], "gray__feature": [1.5]}).to_csv(
            self.features, index=False
        )
        pd.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "status": ["ok", "empty_mask", "upstream_error"],
                "error": ["", "", "failed segmentation"],
            }
        ).to_csv(self.status, index=False)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_valid_contract_preserves_attrition(self) -> None:
        summary = validate_radiomics_contract(self.masks, self.features, self.status)
        self.assertEqual(summary["expected_images"], 3)
        self.assertEqual(summary["successful_images"], 1)
        self.assertEqual(summary["failed_images"], 2)

    def test_missing_status_is_rejected(self) -> None:
        statuses = pd.read_csv(self.status).iloc[:2]
        statuses.to_csv(self.status, index=False)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            validate_radiomics_contract(self.masks, self.features, self.status)

    def test_duplicate_feature_id_is_rejected(self) -> None:
        pd.DataFrame(
            {"image_id": ["a", "a"], "gray__feature": [1.0, 2.0]}
        ).to_csv(self.features, index=False)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_radiomics_contract(self.masks, self.features, self.status)


if __name__ == "__main__":
    unittest.main()
