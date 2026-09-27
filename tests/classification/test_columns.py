from __future__ import annotations

import unittest

import pandas as pd

from scd_ml.classification.columns import (
    CLASS_ORDER,
    FORBIDDEN_FEATURE_COLUMNS,
    encode_target,
    feature_columns,
    radiomic_columns,
)

from .synthetic import synthetic_cohort


class ColumnContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = list(synthetic_cohort().columns)

    def test_channel_sets_select_exactly_their_prefixes(self) -> None:
        gray = radiomic_columns(self.columns, "gray")
        rgb = radiomic_columns(self.columns, "rgb")
        every = radiomic_columns(self.columns, "all")

        self.assertTrue(all(column.startswith("gray__") for column in gray))
        self.assertFalse(any(column.startswith("gray__") for column in rgb))
        self.assertEqual(set(every), set(gray) | set(rgb))

    def test_identifiers_targets_and_acquisition_fields_never_become_features(self) -> None:
        for use_metadata in (False, True):
            radiomics, metadata = feature_columns(
                self.columns, "all", use_metadata=use_metadata
            )
            self.assertFalse(set(radiomics + metadata) & FORBIDDEN_FEATURE_COLUMNS)
        _, metadata = feature_columns(self.columns, "all", use_metadata=True)
        self.assertEqual(metadata, ["age_approx", "sex", "anatom_site_1"])

    def test_target_encoding_follows_class_order_and_rejects_unknown(self) -> None:
        codes = encode_target(pd.Series(list(reversed(CLASS_ORDER))))
        self.assertEqual(codes.tolist(), [3, 2, 1, 0])
        with self.assertRaises(ValueError):
            encode_target(pd.Series(["Benign-melanocytic", "unknown"]))
        with self.assertRaises(ValueError):
            encode_target(pd.Series(["Benign-melanocytic", None]))


if __name__ == "__main__":
    unittest.main()
