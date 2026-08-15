from __future__ import annotations

import json
import unittest
from pathlib import Path


class NotebookLeakageTests(unittest.TestCase):
    def test_isic_notebook_has_no_target_conditioned_imputation(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "notebooks"
            / "eda"
            / "01 ISIC archive - Images.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("age_mode", source)
        self.assertNotIn("sex_mode", source)
        self.assertIn("df_model_input = df_selected.clone()", source)

    def test_global_preprocessing_prototype_does_not_persist_outputs(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "notebooks"
            / "preprocessing"
            / "01 dataset preprocessing.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("clean.to_csv", source)
        self.assertNotIn("dropped_manifest.to_csv", source)


if __name__ == "__main__":
    unittest.main()
