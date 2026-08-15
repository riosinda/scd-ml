from __future__ import annotations

import unittest

from scd_ml.segmentation.training import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_maximizes_validation_dice_and_tracks_best_epoch(self) -> None:
        saved = []
        stopper = EarlyStopping(patience=3, min_delta=1e-4)
        for epoch, score in enumerate([0.50, 0.50005, 0.61, 0.60], start=1):
            stopper.step(score, epoch, lambda e=epoch: saved.append(e))
        self.assertEqual(saved, [1, 3])
        self.assertEqual(stopper.best_epoch, 3)
        self.assertEqual(stopper.best_score, 0.61)

    def test_patience_stops_after_consecutive_non_improvements(self) -> None:
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        self.assertFalse(stopper.step(0.7, 1, lambda: None))
        self.assertFalse(stopper.step(0.6, 2, lambda: None))
        self.assertTrue(stopper.step(0.5, 3, lambda: None))


if __name__ == "__main__":
    unittest.main()
