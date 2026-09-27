from __future__ import annotations

import math
import unittest

from scd_ml.classification.stats import corrected_paired_ttest, friedman_test


class CorrectedPairedTTestTests(unittest.TestCase):
    def test_matches_hand_computed_nadeau_bengio_statistic(self) -> None:
        a = [0.80, 0.82, 0.78, 0.81, 0.79]
        b = [0.78, 0.80, 0.77, 0.78, 0.78]
        # differences 0.02, 0.02, 0.01, 0.03, 0.01 -> mean 0.018, sample var 0.00007
        expected_t = 0.018 / math.sqrt((1 / 5 + 12_000 / 48_000) * 0.00007)

        result = corrected_paired_ttest(a, b, n_train=48_000, n_test=12_000)

        self.assertAlmostEqual(result["mean_difference"], 0.018)
        self.assertAlmostEqual(result["t_statistic"], expected_t, places=6)
        self.assertLess(result["p_value"], 0.05)

    def test_correction_is_more_conservative_than_plain_paired_ttest(self) -> None:
        a = [0.80, 0.82, 0.78, 0.81, 0.79]
        b = [0.79, 0.815, 0.775, 0.80, 0.785]
        corrected = corrected_paired_ttest(a, b, n_train=48_000, n_test=12_000)
        plain = corrected_paired_ttest(a, b, n_train=1e12, n_test=1)
        self.assertGreater(corrected["p_value"], plain["p_value"])

    def test_identical_scores_are_not_significant(self) -> None:
        result = corrected_paired_ttest([0.5] * 5, [0.5] * 5, n_train=4, n_test=1)
        self.assertEqual(result["p_value"], 1.0)

    def test_friedman_requires_three_models(self) -> None:
        with self.assertRaises(ValueError):
            friedman_test({"a": [1, 2], "b": [2, 3]})
        result = friedman_test({"a": [0.5, 0.6, 0.7], "b": [0.4, 0.5, 0.6], "c": [0.3, 0.4, 0.5]})
        self.assertLess(result["p_value"], 0.1)


if __name__ == "__main__":
    unittest.main()
