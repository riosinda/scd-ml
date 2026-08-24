from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scd_ml.classification.preprocessing import FoldLocalRadiomicsTransformer


class FoldLocalRadiomicsTransformerTests(unittest.TestCase):
    def test_statistics_and_correlation_filter_are_learned_from_train_only(self) -> None:
        training = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                "c": [1.0, np.inf, 2.0, 5.0, 3.0],
                "empty": [np.nan] * 5,
            }
        )
        validation = pd.DataFrame(
            {
                "a": [100.0, -100.0],
                "b": [1.0, 7.0],
                "c": [np.nan, 4.0],
                "empty": [9.0, 9.0],
            }
        )

        transformer = FoldLocalRadiomicsTransformer(
            winsor_low=0.0,
            winsor_high=1.0,
            correlation_threshold=0.95,
        ).fit(training)
        transformed = transformer.transform(validation)

        self.assertEqual(transformer.all_missing_features_, ["empty"])
        self.assertNotIn("a", transformed.columns)
        self.assertIn("b", transformed.columns)
        self.assertEqual(transformer.medians_["c"], 2.5)
        self.assertEqual(transformed.loc[0, "c"], 2.5)
        self.assertEqual(transformed.loc[1, "b"], 7.0)

    def test_missing_training_feature_is_rejected_at_transform_time(self) -> None:
        training = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
        transformer = FoldLocalRadiomicsTransformer().fit(training)
        with self.assertRaisesRegex(ValueError, "missing features"):
            transformer.transform(pd.DataFrame({"a": [1.0]}))


if __name__ == "__main__":
    unittest.main()
