from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scd_ml.classification.tracking import Tracker, flatten, log_cv_child, numeric_metrics


def fold_scores(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": range(5),
            "f1_macro": [0.5 + offset + 0.01 * fold for fold in range(5)],
            "balanced_accuracy": [0.6] * 5,
        }
    )


class TrackingHelperTests(unittest.TestCase):
    def test_flatten_and_numeric_metrics(self) -> None:
        self.assertEqual(
            flatten({"a": 1, "b": {"c": [1, 2]}}), {"a": "1", "b.c": "[1, 2]"}
        )
        self.assertEqual(
            numeric_metrics({"x": 1, "flag": True, "name": "gray", "none": None}, prefix="m_"),
            {"m_x": 1.0},
        )

    def test_disabled_tracker_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = Tracker(
                enabled=False, experiment="x", output_dir=Path(directory), run_name="x"
            )
            with tracker:
                log_cv_child(tracker, "config", fold_scores(), params={"a": 1})
                tracker.log_metrics({"m": 1.0})
            self.assertEqual(list(Path(directory).iterdir()), [])


class MlflowTrackerTests(unittest.TestCase):
    def test_resumed_parent_logs_each_child_once_and_replaces_new_versions(self) -> None:
        from mlflow.tracking import MlflowClient

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uri = f"sqlite:///{root / 'mlflow' / 'mlflow.db'}"
            output = root / "stage"
            output.mkdir()

            def open_tracker() -> Tracker:
                return Tracker(
                    enabled=True,
                    experiment="classification-test-suite",
                    output_dir=output,
                    run_name="stage",
                    tracking_uri=uri,
                    params={"seed": 42, "strategy": {"selection": "anova"}},
                )

            with open_tracker() as tracker:
                parent = tracker.run_id
                log_cv_child(tracker, "a", fold_scores(), params={"model": "logreg"})
                log_cv_child(tracker, "b", fold_scores(), params={"model": "xgboost"})
            with open_tracker() as tracker:
                self.assertEqual(tracker.run_id, parent)
                log_cv_child(tracker, "a", fold_scores(), params={"model": "logreg"})
                log_cv_child(
                    tracker, "b", fold_scores(0.1), params={"model": "xgboost"}, version="v2"
                )

            client = MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name("classification-test-suite")
            children = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent}'",
            )
            by_key = {run.data.tags["scd.key"]: run for run in children}
            self.assertEqual(sorted(by_key), ["a", "b"])
            self.assertAlmostEqual(by_key["b"].data.metrics["f1_macro_mean"], 0.62)
            history = client.get_metric_history(by_key["a"].info.run_id, "fold_f1_macro")
            self.assertEqual(sorted(metric.step for metric in history), [0, 1, 2, 3, 4])
            parent_run = client.get_run(parent)
            self.assertEqual(parent_run.data.params["strategy.selection"], "anova")
            self.assertEqual(parent_run.info.status, "FINISHED")


if __name__ == "__main__":
    unittest.main()
