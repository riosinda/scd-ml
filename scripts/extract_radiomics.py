#!/usr/bin/env python3
"""Extract radiomics from the segmentation manifest using Python 3.7.17.

This file is deliberately standalone: it must not import ``scd_ml`` or any
module that targets Python 3.12. Its only cross-runtime interface is CSV.
"""

import argparse
import csv
import gc
import logging
import multiprocessing as mp
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import radiomics
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASKS_MANIFEST = ROOT / "results" / "segmentation" / "isic_masks_manifest.csv"
DEFAULT_FEATURES = ROOT / "results" / "features" / "radiomics_features.csv"
DEFAULT_STATUS = ROOT / "results" / "features" / "radiomics_status.csv"
DEFAULT_CONFIG = ROOT / "configs" / "radiomics.yaml"
SUCCESS_UPSTREAM = {"segmented", "no_detection"}
BATCH_SIZE = 200

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
radiomics.setVerbosity(logging.ERROR)
_extractor = None


def _worker_init(config_path):
    global _extractor
    _extractor = featureextractor.RadiomicsFeatureExtractor(str(config_path))


def _to_scalar(value):
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    raise ValueError("Radiomics returned a non-scalar feature value")


def _extract_one(row):
    image_id = str(row["image_id"])
    try:
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot_read_image")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("cannot_read_mask")
        if image.shape[:2] != mask.shape:
            raise ValueError("image_mask_dimension_mismatch")
        if not np.any(mask > 127):
            return None, {"image_id": image_id, "status": "empty_mask", "error": ""}

        mask_sitk = sitk.GetImageFromArray((mask > 127).astype(np.uint8))
        channels = {
            "blue": image[:, :, 0],
            "green": image[:, :, 1],
            "red": image[:, :, 2],
            "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }
        output = {"image_id": image_id}
        for channel_name, channel in channels.items():
            raw = _extractor.execute(sitk.GetImageFromArray(channel), mask_sitk)
            for key, value in raw.items():
                if not key.startswith("diagnostics_"):
                    output["{}__{}".format(channel_name, key)] = _to_scalar(value)
        return output, {"image_id": image_id, "status": "ok", "error": ""}
    except Exception:
        error = traceback.format_exc().strip().replace("\n", " | ")
        return None, {"image_id": image_id, "status": "error", "error": error}


def _flush_features(buffer, output_path):
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )
    buffer[:] = []
    gc.collect()


def _read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "image_path", "mask_path", "status"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError("Mask manifest is missing columns: {}".format(sorted(missing)))
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Mask manifest image_id values must be unique")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-manifest", type=Path, default=DEFAULT_MASKS_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.overwrite and (args.features.exists() or args.status.exists()):
        raise FileExistsError("Radiomics output exists; pass --overwrite")
    if args.overwrite and args.features.exists():
        args.features.unlink()
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(args.masks_manifest)
    eligible = []

    with args.status.open("w", newline="", encoding="utf-8") as status_handle:
        status_writer = csv.DictWriter(status_handle, fieldnames=["image_id", "status", "error"])
        status_writer.writeheader()
        for row in rows:
            if row["status"] in SUCCESS_UPSTREAM:
                eligible.append(row)
            else:
                status_writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "status": "upstream_{}".format(row["status"]),
                        "error": row.get("error", ""),
                    }
                )

        feature_buffer = []
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(args.config,)) as pool:
            iterator = pool.imap_unordered(_extract_one, eligible, chunksize=4)
            for features, status in tqdm(iterator, total=len(eligible), desc="Radiomics"):
                status_writer.writerow(status)
                status_handle.flush()
                if features is not None:
                    feature_buffer.append(features)
                if len(feature_buffer) >= BATCH_SIZE:
                    _flush_features(feature_buffer, args.features)
        _flush_features(feature_buffer, args.features)

    if not args.features.exists():
        args.features.write_text("image_id\n", encoding="utf-8")
    print("Features: {}".format(args.features))
    print("Status:   {}".format(args.status))


if __name__ == "__main__":
    main()
