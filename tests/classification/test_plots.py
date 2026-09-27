from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from scd_ml.classification.columns import CLASS_ORDER
from scd_ml.classification.metrics import per_class_report
from scd_ml.classification.plots import (
    feature_family,
    holdout_figures,
    screening_figures,
    tuning_figures,
)
from scd_ml.classification.screening import feature_stability, rank_strategies


def screening_summary() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for channel_set in ("all", "gray"):
        for selection in ("none", "anova"):
            for balancing in ("none", "smote"):
                for model in ("logreg", "xgboost"):
                    rows.append(
                        {
                            "config": f"{channel_set}|{selection}|{balancing}|{model}",
                            "channel_set": channel_set,
                            "selection": selection,
                            "balancing": balancing,
                            "model": model,
                            "f1_macro_mean": rng.uniform(0.4, 0.7),
                        }
                    )
    return pd.DataFrame(rows)


def selected_features() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for feature in ("gray__original_glcm_Contrast", f"gray__original_firstorder_F{fold}"):
            rows.append(
                {
                    "config": "c",
                    "channel_set": "gray",
                    "selection": "anova",
                    "fold": fold,
                    "feature": feature,
                }
            )
    return pd.DataFrame(rows)


class PlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.rc = dict(matplotlib.rcParams)

    def tearDown(self) -> None:
        # Figures are drawn inside rc_context, so global style must be untouched.
        self.assertEqual(dict(matplotlib.rcParams), self.rc)
        self.directory.cleanup()

    def assert_files(self, paths: list[Path], expected: set[str]) -> None:
        self.assertEqual({path.stem for path in paths}, expected)
        for path in paths:
            self.assertGreater(path.stat().st_size, 0)

    def test_feature_family_parses_radiomics_and_flags_metadata(self) -> None:
        self.assertEqual(feature_family("red__original_glszm_ZoneEntropy"), "glszm")
        self.assertEqual(feature_family("gray__original_shape2D_Perimeter"), "shape2D")
        self.assertEqual(feature_family("sex__male"), "metadata")
        self.assertEqual(feature_family("age_approx"), "metadata")

    def test_screening_figures_include_feature_selection(self) -> None:
        summary = screening_summary()
        frequency, jaccard = feature_stability(selected_features())
        paths = screening_figures(
            summary, rank_strategies(summary), self.path, frequency=frequency, jaccard=jaccard
        )
        self.assert_files(
            paths,
            {
                "strategy_ranking",
                "screening_f1_heatmap",
                "feature_selection_stability",
                "selected_features_gray",
            },
        )

    def test_tuning_figures(self) -> None:
        summary = pd.DataFrame(
            {
                "study": ["gray_logreg", "all_logreg", "all_logreg_meta"],
                "channel_set": ["gray", "all", "all"],
                "use_metadata": [False, False, True],
                "model": ["logreg"] * 3,
                "k": [20.0, 30.0, np.nan],
                "f1_macro_mean": [0.5, 0.55, 0.6],
                "f1_macro_std": [0.02, 0.03, 0.02],
                **{f"recall__{name}_mean": [0.4, 0.5, 0.6] for name in CLASS_ORDER},
            }
        )
        channels = summary.iloc[:2].assign(reference="gray", p_value_vs_reference=[np.nan, 0.01])
        metadata = pd.DataFrame(
            {
                "model": ["logreg"],
                "channel_set": ["all"],
                "f1_macro_radiomics": [0.55],
                "f1_macro_with_metadata": [0.6],
                "p_value": [0.04],
            }
        )
        history = pd.DataFrame(
            {
                "study": ["gray_logreg"] * 3,
                "model": ["logreg"] * 3,
                "number": [0, 1, 2],
                "value": [0.4, 0.5, 0.45],
            }
        )
        frequency, _ = feature_stability(selected_features())
        paths = tuning_figures(
            summary,
            self.path,
            channel_ablation=channels,
            metadata_ablation=metadata,
            history=history,
            winner_selection=frequency,
            winner="gray_logreg",
        )
        self.assert_files(
            paths,
            {
                "studies_f1",
                "per_class_recall",
                "channel_ablation",
                "metadata_ablation",
                "optimization_history",
                "winner_selected_features",
            },
        )

    def test_holdout_figures(self) -> None:
        rng = np.random.default_rng(0)
        y = np.arange(40) % len(CLASS_ORDER)
        proba = rng.dirichlet(np.ones(len(CLASS_ORDER)), size=len(y))
        proba[np.arange(len(y)), y] += 1
        proba /= proba.sum(axis=1, keepdims=True)
        per_class = per_class_report(y, proba.argmax(axis=1))
        selected = ["gray__original_glcm_Idm", "sex__male"]
        paths = holdout_figures(y, proba, per_class, self.path, selected_features=selected)
        self.assert_files(
            paths,
            {"confusion_matrix", "roc_pr_curves", "per_class_metrics", "final_selected_features"},
        )


if __name__ == "__main__":
    unittest.main()
