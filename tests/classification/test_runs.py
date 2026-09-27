from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from scd_ml.classification.runs import prepare_run_dir, write_json


def load_evaluate_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_classifier.py"
    spec = importlib.util.spec_from_file_location("evaluate_classifier_for_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunDirectoryTests(unittest.TestCase):
    def test_resume_requires_the_same_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.assertFalse(prepare_run_dir(output, {"folds": (0, 1)}, overwrite=False))
            (output / "progress.csv").write_text("x\n")
            self.assertTrue(prepare_run_dir(output, {"folds": [0, 1]}, overwrite=False))
            with self.assertRaisesRegex(ValueError, "different configuration"):
                prepare_run_dir(output, {"folds": [0]}, overwrite=False)

            self.assertFalse(prepare_run_dir(output, {"folds": [0]}, overwrite=True))
            self.assertFalse((output / "progress.csv").exists())

    def test_foreign_non_empty_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "other.csv").write_text("x\n")
            with self.assertRaises(FileExistsError):
                prepare_run_dir(Path(directory), {}, overwrite=False)


class FrozenTestGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_evaluate_module()

    def test_missing_or_incomplete_winner_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tuning = Path(directory)
            with self.assertRaises(FileNotFoundError):
                self.module.load_winner(tuning)
            write_json(tuning / "winner.json", {"complete": False, "folds": [0, 1, 2, 3, 4]})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.module.load_winner(tuning)
            write_json(tuning / "winner.json", {"complete": True, "folds": [0, 1]})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.module.load_winner(tuning)
            write_json(tuning / "winner.json", {"complete": True, "folds": [0, 1, 2, 3, 4]})
            self.assertTrue(self.module.load_winner(tuning)["complete"])

    def test_existing_test_results_require_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test"
            model = Path(directory) / "winner.joblib"
            self.module.guard_outputs(output, model, overwrite=False)
            output.mkdir()
            (output / "metrics.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                self.module.guard_outputs(output, model, overwrite=False)
            self.module.guard_outputs(output, model, overwrite=True)


if __name__ == "__main__":
    unittest.main()
