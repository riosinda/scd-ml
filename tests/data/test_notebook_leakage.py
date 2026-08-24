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

    def test_classification_datasets_fit_preprocessing_on_fold_train_only(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "notebooks"
            / "preprocessing"
            / "02 fold-local classification datasets.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn(
            "transformer.fit_transform(train_rows[radiomic_columns])", source
        )
        self.assertIn(
            "transformer.transform(validation_rows[radiomic_columns])", source
        )
        self.assertNotIn("transformer.fit_transform(cohort", source)
        self.assertNotIn("transformer.transform(eligible_test", source)
        self.assertIn("metadata_columns = list(DEFAULT_METADATA_COLUMNS)", source)
        self.assertIn("train_rows[id_columns + metadata_columns]", source)
        self.assertIn("validation_rows[id_columns + metadata_columns]", source)
        self.assertIn(
            "][id_columns + metadata_columns + radiomic_columns].copy()", source
        )
        self.assertIn("cohort_coverage_by_class.csv", source)


if __name__ == "__main__":
    unittest.main()
