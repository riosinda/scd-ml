from __future__ import annotations

import unittest

import pandas as pd

from scd_ml.classification.cv import FoldCache, evaluate_config, fold_scores_frame
from scd_ml.classification.datasets import split_development_fold
from scd_ml.classification.pipeline import ModelConfig
from scd_ml.classification.reporting import channel_ablation, metadata_ablation
from scd_ml.classification.screening import feature_stability, rank_strategies

from .synthetic import synthetic_cohort


class FoldCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = synthetic_cohort()
        self.cache = FoldCache(self.cohort, preprocessing={"correlation_threshold": 0.99})

    def test_head_statistics_come_from_fold_training_rows_only(self) -> None:
        for data in self.cache.get("gray", use_metadata=True):
            training, validation = split_development_fold(self.cohort, data.fold)
            radiomics = data.head.named_steps["columns"].named_transformers_["radiomics"]
            expected = training[radiomics.medians_.index].median()
            pd.testing.assert_series_equal(radiomics.medians_, expected, check_names=False)
            self.assertFalse(set(training["group_id"]) & set(validation["group_id"]))
            self.assertEqual(data.valid_image_ids.tolist(), validation["image_id"].tolist())

    def test_heads_are_reused_per_channel_set_and_metadata(self) -> None:
        self.assertIs(self.cache.get("gray", False), self.cache.get("gray", False))
        self.assertIsNot(self.cache.get("gray", False), self.cache.get("gray", True))

    def test_out_of_fold_predictions_cover_eligible_development_exactly_once(self) -> None:
        config = ModelConfig("all", False, "anova", "class_weight", "logreg", k=5)
        results = evaluate_config(config, self.cache.get("all", False), fold_jobs=2)
        oof = pd.concat([result.oof for result in results])
        eligible = self.cohort[
            self.cohort["eligible_for_classification"] & self.cohort["split"].eq("train")
        ]

        self.assertEqual(sorted(oof["image_id"]), sorted(eligible["image_id"]))
        self.assertTrue(oof["image_id"].is_unique)
        scores = fold_scores_frame(config, results)
        self.assertEqual(scores["fold"].tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(scores["n_selected_features"].eq(5).all())


class ReportingTests(unittest.TestCase):
    def test_strategy_ranking_averages_ranks_over_cells(self) -> None:
        summary = pd.DataFrame(
            {
                "channel_set": ["gray", "gray", "rgb", "rgb"],
                "model": ["logreg"] * 4,
                "selection": ["anova", "none", "anova", "none"],
                "balancing": ["smote"] * 4,
                "f1_macro_mean": [0.60, 0.55, 0.50, 0.58],
            }
        )
        ranking = rank_strategies(summary)
        self.assertEqual(ranking["mean_rank"].tolist(), [1.5, 1.5])
        self.assertEqual(ranking.iloc[0]["selection"], "none")

    def test_feature_stability_reports_frequency_and_jaccard(self) -> None:
        selected = pd.DataFrame(
            {
                "config": ["c"] * 4,
                "channel_set": ["gray"] * 4,
                "selection": ["anova"] * 4,
                "fold": [0, 0, 1, 1],
                "feature": ["a", "b", "a", "c"],
            }
        )
        frequency, jaccard = feature_stability(selected)
        self.assertEqual(frequency.set_index("feature")["n_folds_selected"]["a"], 2)
        self.assertAlmostEqual(jaccard["jaccard"].iloc[0], 1 / 3)

    def test_ablation_tables_pair_studies_by_fold(self) -> None:
        rows = []
        for study, channel_set, use_metadata, offset in [
            ("gray_logreg", "gray", False, 0.00),
            ("all_logreg", "all", False, 0.05),
            ("all_logreg_meta", "all", True, 0.07),
        ]:
            for fold in range(5):
                rows.append(
                    {
                        "study": study,
                        "channel_set": channel_set,
                        "use_metadata": use_metadata,
                        "model": "logreg",
                        "fold": fold,
                        "n_train": 80,
                        "n_valid": 20,
                        "f1_macro": 0.5 + offset + 0.01 * fold * (1 + offset),
                    }
                )
        fold_scores = pd.DataFrame(rows)

        channels = channel_ablation(fold_scores).set_index("channel_set")
        self.assertGreater(channels.loc["all", "mean_difference_vs_reference"], 0)
        metadata = metadata_ablation(fold_scores, "all")
        self.assertAlmostEqual(metadata["mean_difference"].iloc[0], 0.0204, 6)


if __name__ == "__main__":
    unittest.main()
