from __future__ import annotations

import unittest

import numpy as np

from scd_ml.segmentation.metrics import binary_segmentation_metrics


class SegmentationMetricTests(unittest.TestCase):
    def test_no_prediction_is_not_perfect_precision(self) -> None:
        pred = np.zeros((2, 2), dtype=bool)
        truth = np.array([[1, 0], [0, 0]], dtype=bool)
        metrics = binary_segmentation_metrics(pred, truth)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)

    def test_false_positive_has_zero_precision(self) -> None:
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        truth = np.zeros((2, 2), dtype=bool)
        metrics = binary_segmentation_metrics(pred, truth)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)

    def test_perfect_prediction(self) -> None:
        mask = np.array([[1, 1], [0, 0]], dtype=bool)
        metrics = binary_segmentation_metrics(mask, mask)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)

    def test_both_empty_has_defined_overlap_but_zero_precision(self) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        metrics = binary_segmentation_metrics(mask, mask)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
