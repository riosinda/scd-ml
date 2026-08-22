from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def load_extractor_module():
    """Load the Python 3.7 extractor without installing its heavy runtime."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "extract_radiomics.py"

    cv2 = types.ModuleType("cv2")
    simple_itk = types.ModuleType("SimpleITK")
    simple_itk.ProcessObject = types.SimpleNamespace(
        SetGlobalDefaultNumberOfThreads=lambda _threads: None
    )
    radiomics = types.ModuleType("radiomics")
    radiomics.setVerbosity = lambda _level: None
    radiomics.featureextractor = types.SimpleNamespace()
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterator, **_kwargs: iterator

    stubs = {
        "cv2": cv2,
        "SimpleITK": simple_itk,
        "radiomics": radiomics,
        "tqdm": tqdm_module,
    }
    spec = importlib.util.spec_from_file_location("extract_radiomics_for_test", script)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FakePool:
    processed_ids = []

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def imap_unordered(self, _function, rows, chunksize):
        self.__class__.processed_ids = [str(row["image_id"]) for row in rows]
        assert chunksize == 4
        for image_id in self.__class__.processed_ids:
            yield (
                {"image_id": image_id, "gray__feature": 2.5},
                {"image_id": image_id, "status": "ok", "error": ""},
            )


class RadiomicsResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_extractor_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.manifest = self.root / "masks.csv"
        self.features = self.root / "features.csv"
        self.status = self.root / "status.csv"
        self.config = self.root / "radiomics.yaml"
        self.config.write_text("imageType: {}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_main(self) -> None:
        argv = [
            "extract_radiomics.py",
            "--masks-manifest",
            str(self.manifest),
            "--features",
            str(self.features),
            "--status",
            str(self.status),
            "--config",
            str(self.config),
            "--workers",
            "1",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            self.module.mp, "Pool", FakePool
        ):
            self.module.main()

    def test_resume_preserves_features_and_only_retries_incomplete_ids(self) -> None:
        pd.DataFrame(
            {
                "image_id": ["done", "retry", "empty", "upstream"],
                "image_path": ["done.jpg", "retry.jpg", "empty.jpg", "missing.jpg"],
                "mask_path": ["done.png", "retry.png", "empty.png", ""],
                "status": ["segmented", "segmented", "no_detection", "error"],
            }
        ).to_csv(self.manifest, index=False)
        pd.DataFrame({"image_id": ["done"], "gray__feature": [1.5]}).to_csv(
            self.features, index=False
        )
        pd.DataFrame(
            {
                "image_id": ["done", "retry", "empty", "upstream"],
                "status": ["ok", "error", "empty_mask", "upstream_error"],
                "error": ["", "interrupted", "", "missing image"],
            }
        ).to_csv(self.status, index=False)

        self.run_main()

        self.assertEqual(FakePool.processed_ids, ["retry"])
        features = pd.read_csv(self.features).set_index("image_id")
        self.assertEqual(set(features.index), {"done", "retry"})
        self.assertEqual(features.loc["done", "gray__feature"], 1.5)
        statuses = pd.read_csv(self.status).set_index("image_id")
        self.assertEqual(statuses.loc["done", "status"], "ok")
        self.assertEqual(statuses.loc["retry", "status"], "ok")
        self.assertEqual(statuses.loc["empty", "status"], "empty_mask")
        self.assertEqual(statuses.loc["upstream", "status"], "upstream_error")
        self.assertFalse(Path(str(self.status) + ".resume").exists())

    def test_duplicate_existing_feature_ids_are_rejected(self) -> None:
        self.features.write_text(
            "image_id,gray__feature\nduplicate,1\nduplicate,2\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate image_id"):
            self.module._read_feature_ids(self.features)


if __name__ == "__main__":
    unittest.main()
