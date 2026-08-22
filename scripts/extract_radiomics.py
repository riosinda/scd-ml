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
STATUS_FIELDS = ["image_id", "status", "error"]

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
        header=not output_path.exists() or output_path.stat().st_size == 0,
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


def _read_feature_ids(path):
    """Read only image IDs so a large feature CSV can be resumed in constant RAM."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError("Existing feature CSV is missing the image_id column")
        image_ids = set()
        for row in reader:
            image_id = str(row.get("image_id", ""))
            if not image_id:
                raise ValueError("Existing feature CSV contains an empty image_id")
            if image_id in image_ids:
                raise ValueError(
                    "Existing feature CSV contains duplicate image_id: {}".format(image_id)
                )
            image_ids.add(image_id)
    return image_ids


def _read_statuses(path):
    """Return the latest status for every ID; journals may contain retries."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(STATUS_FIELDS)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Existing status CSV is missing columns: {}".format(
                    sorted(required - set(reader.fieldnames or []))
                )
            )
        return {
            str(row["image_id"]): {
                "image_id": str(row["image_id"]),
                "status": row["status"],
                "error": row.get("error", ""),
            }
            for row in reader
            if row.get("image_id")
        }


def _write_final_status(path, rows, statuses):
    """Atomically compact the base status and resume journal in manifest order."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            image_id = str(row["image_id"])
            if image_id not in statuses:
                raise RuntimeError("Missing final radiomics status for {}".format(image_id))
            writer.writerow(statuses[image_id])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _unlink_if_exists(path):
    if path.exists():
        path.unlink()


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
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    journal_path = args.status.with_name(args.status.name + ".resume")
    if args.overwrite:
        _unlink_if_exists(args.features)
        _unlink_if_exists(args.status)
        _unlink_if_exists(journal_path)

    rows = _read_manifest(args.masks_manifest)
    manifest_ids = {str(row["image_id"]) for row in rows}
    feature_ids = _read_feature_ids(args.features)
    unexpected_features = feature_ids - manifest_ids
    if unexpected_features:
        raise ValueError(
            "Existing features do not belong to this mask manifest; first unexpected IDs: {}. "
            "Use the matching manifest or pass --overwrite.".format(
                sorted(unexpected_features)[:5]
            )
        )

    previous_statuses = _read_statuses(args.status)
    previous_statuses.update(_read_statuses(journal_path))
    statuses = {}
    eligible = []
    reused_empty_masks = 0
    retried = 0

    for row in rows:
        image_id = str(row["image_id"])
        if row["status"] not in SUCCESS_UPSTREAM:
            if image_id in feature_ids:
                raise ValueError(
                    "Existing features for {} conflict with upstream status {!r}; "
                    "use the matching manifest or pass --overwrite.".format(
                        image_id, row["status"]
                    )
                )
            statuses[image_id] = {
                "image_id": image_id,
                "status": "upstream_{}".format(row["status"]),
                "error": row.get("error", ""),
            }
        elif image_id in feature_ids:
            # The feature row is the authoritative durable checkpoint. A prior
            # status may have been lost if the process stopped during compaction.
            statuses[image_id] = {"image_id": image_id, "status": "ok", "error": ""}
        elif previous_statuses.get(image_id, {}).get("status") == "empty_mask":
            statuses[image_id] = previous_statuses[image_id]
            reused_empty_masks += 1
        else:
            if image_id in previous_statuses:
                retried += 1
            eligible.append(row)

    print("Manifest rows:        {:,}".format(len(rows)))
    print("Existing features:    {:,}".format(len(feature_ids)))
    print("Existing empty masks: {:,}".format(reused_empty_masks))
    print("Retries:              {:,}".format(retried))
    print("Pending extraction:   {:,}".format(len(eligible)))

    journal_has_rows = journal_path.exists() and journal_path.stat().st_size > 0
    with journal_path.open("a", newline="", encoding="utf-8") as status_handle:
        status_writer = csv.DictWriter(status_handle, fieldnames=STATUS_FIELDS)
        if not journal_has_rows:
            status_writer.writeheader()
            status_handle.flush()

        feature_buffer = []
        results_since_checkpoint = 0
        if eligible:
            with mp.Pool(
                args.workers, initializer=_worker_init, initargs=(args.config,)
            ) as pool:
                iterator = pool.imap_unordered(_extract_one, eligible, chunksize=4)
                for features, status in tqdm(
                    iterator, total=len(eligible), desc="Radiomics"
                ):
                    statuses[str(status["image_id"])] = status
                    status_writer.writerow(status)
                    if features is not None:
                        feature_buffer.append(features)
                    results_since_checkpoint += 1
                    if len(feature_buffer) >= BATCH_SIZE:
                        _flush_features(feature_buffer, args.features)
                    if results_since_checkpoint >= BATCH_SIZE:
                        status_handle.flush()
                        results_since_checkpoint = 0
            _flush_features(feature_buffer, args.features)
        status_handle.flush()

    if not args.features.exists() or args.features.stat().st_size == 0:
        args.features.write_text("image_id\n", encoding="utf-8")
    _write_final_status(args.status, rows, statuses)
    _unlink_if_exists(journal_path)
    print("Features: {}".format(args.features))
    print("Status:   {}".format(args.status))


if __name__ == "__main__":
    main()
