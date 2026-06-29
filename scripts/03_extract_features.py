"""Extract pyradiomics features from ISIC images + Mask R-CNN masks.

Reads matched image/mask pairs, extracts R/G/B/gray radiomic features per
image using N_WORKERS parallel processes, and streams results to
results/features/features_all_channels.csv (resumes if interrupted).
Finally converts the CSV to Parquet.

Usage (from repo root, with .venv active):
    python scripts/extract_features.py              # auto workers (cpu_count - 2)
    python scripts/extract_features.py --workers 28 # explicit

Run inside tmux:
    tmux new -s features
    python scripts/extract_features.py 2>&1 | tee features.log
"""
from __future__ import annotations

# Thread limits must be set BEFORE importing numpy/SimpleITK/radiomics in every
# process — worker processes inherit this via fork on Linux.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import argparse
import gc
import logging
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import radiomics
from radiomics import featureextractor

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
radiomics.setVerbosity(logging.ERROR)

from src.paths import ISIC_IMAGES_DIR, ISIC_MASKS_DIR, RESULTS_DIR, ensure_dir

FEATURES_DIR = ensure_dir(RESULTS_DIR / "features")
OUT_CSV      = FEATURES_DIR / "features_all_channels.csv"
OUT_PARQUET  = FEATURES_DIR / "features_all_channels.parquet"
LOG_PATH     = FEATURES_DIR / "errors.log"
BATCH_SIZE   = 200
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".tif")

# ─────────────── Worker (one extractor per process) ─────────────────────────

_extractor: featureextractor.RadiomicsFeatureExtractor | None = None


def _worker_init():
    global _extractor
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    for cls in ["firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm", "shape2D"]:
        ext.enableFeatureClassByName(cls)
    _extractor = ext


def _to_scalar(value):
    if isinstance(value, (list, tuple, set)):
        return next(iter(value))
    if isinstance(value, dict):
        return next(iter(value.values()))
    return value


def _extract_one(args: tuple[str, Path, Path]) -> tuple[str, str, dict | str]:
    """Return ("ok", image_id, features) or ("err", image_id, traceback_str)."""
    image_id, img_path, msk_path = args
    try:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")
        msk_np = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if msk_np is None:
            raise ValueError(f"Cannot read mask: {msk_path}")
        if msk_np.max() == 0:
            raise ValueError("Empty mask (all zeros)")

        msk_sitk = sitk.GetImageFromArray((msk_np > 127).astype(np.uint8))
        channels = {
            "blue":  img_bgr[:, :, 0],
            "green": img_bgr[:, :, 1],
            "red":   img_bgr[:, :, 2],
            "gray":  cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        }
        out: dict = {}
        for ch_name, arr in channels.items():
            raw = _extractor.execute(sitk.GetImageFromArray(arr), msk_sitk)
            for k, v in raw.items():
                if "diagnostics_" not in k:
                    out[f"{ch_name}__{k}"] = _to_scalar(v)
        out["image_id"] = image_id
        return "ok", image_id, out
    except Exception:
        return "err", image_id, traceback.format_exc()


# ─────────────── Helpers ────────────────────────────────────────────────────

def find_pairs() -> tuple[dict[str, str], dict[str, str], list[str]]:
    image_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(ISIC_IMAGES_DIR)
        if f.lower().endswith(IMG_EXTS)
    }
    mask_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(ISIC_MASKS_DIR)
        if f.lower().endswith(IMG_EXTS)
    }
    pair_ids = sorted(set(image_files) & set(mask_files))
    print(f"Images:        {len(image_files):,}")
    print(f"Masks:         {len(mask_files):,}")
    print(f"Matched pairs: {len(pair_ids):,}")
    return image_files, mask_files, pair_ids


def get_processed_ids() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    try:
        return set(pd.read_csv(OUT_CSV, usecols=["image_id"])["image_id"].astype(str))
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return set()


def flush(buffer: list[dict]) -> None:
    pd.DataFrame(buffer).set_index("image_id").to_csv(
        OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=True
    )
    buffer.clear()
    gc.collect()


# ─────────────── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 2),
                        help="parallel worker processes (default: cpu_count - 2)")
    args = parser.parse_args()

    print(f"Images:    {ISIC_IMAGES_DIR}")
    print(f"Masks:     {ISIC_MASKS_DIR}")
    print(f"Features → {FEATURES_DIR}")
    print(f"Workers:   {args.workers} / {os.cpu_count()} CPUs\n")

    image_files, mask_files, pair_ids = find_pairs()
    if not pair_ids:
        print("No matched pairs — are the GCS buckets mounted?")
        sys.exit(1)

    processed  = get_processed_ids()
    to_process = [pid for pid in pair_ids if pid not in processed]
    print(f"\nAlready processed: {len(processed):,}")
    print(f"To process:        {len(to_process):,}\n")

    tasks = [
        (pid, ISIC_IMAGES_DIR / image_files[pid], ISIC_MASKS_DIR / mask_files[pid])
        for pid in to_process
    ]

    buffer   = []
    n_ok     = 0
    n_errors = 0

    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        for status, image_id, data in tqdm(
            pool.imap_unordered(_extract_one, tasks, chunksize=4),
            total=len(tasks),
            desc="Extracting",
        ):
            if status == "ok":
                buffer.append(data)
                n_ok += 1
            else:
                n_errors += 1
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"\n--- image_id={image_id} ---\n{data}\n")

            if len(buffer) >= BATCH_SIZE:
                flush(buffer)

    if buffer:
        flush(buffer)

    print(f"\nDone. OK: {n_ok:,}  Errors: {n_errors} (see {LOG_PATH})")

    if OUT_CSV.exists():
        print("\nConverting CSV → Parquet...")
        df = pd.read_csv(OUT_CSV, index_col="image_id")
        df.to_parquet(OUT_PARQUET, compression="snappy")
        print(f"Saved {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1024**2:.1f} MB)")
        print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
