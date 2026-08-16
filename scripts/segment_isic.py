#!/usr/bin/env python3
"""Apply the HAM10000 segmenter to every image in an ISIC split manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as vision_functional
from tqdm import tqdm

from scd_ml.data.schemas import require_columns, require_unique
from scd_ml.paths import ISIC_IMAGES_DIR, ISIC_MASKS_DIR, MODELS_DIR, SEGMENTATION_DIR, SPLITS_DIR
from scd_ml.segmentation.inference import best_binary_mask
from scd_ml.segmentation.model import build_mask_rcnn, load_model_checkpoint

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANIFEST_COLUMNS = [
    "image_id", "image_path", "mask_path", "score", "num_detections", "status", "error"
]


def _log(message: str) -> None:
    print(message, flush=True)


def _index_images(images_dir: Path) -> dict[str, Path]:
    """Index a flat image directory with one GCS FUSE directory listing.

    ``Path.iterdir()`` plus ``Path.is_file()`` can issue a metadata request per
    object on a mounted bucket.  The legacy inference script used
    ``os.listdir()`` and was substantially faster for this flat ISIC layout.
    Individual files are validated when Pillow opens them during inference.
    """
    names = os.listdir(images_dir)
    return {
        Path(name).stem: images_dir / name
        for name in names
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    }


def _find_existing_masks(masks_dir: Path, expected_ids: set[str]) -> list[Path]:
    """Find expected masks with one directory listing instead of one stat per ID."""
    if not masks_dir.exists():
        return []
    return [
        masks_dir / name
        for name in os.listdir(masks_dir)
        if Path(name).suffix.lower() == ".png" and Path(name).stem in expected_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "isic_classification.csv")
    parser.add_argument("--images-dir", type=Path, default=ISIC_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=ISIC_MASKS_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=MODELS_DIR / "segmentation" / "mask_rcnn_best.pt"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    parser.add_argument(
        "--output-manifest", type=Path, default=SEGMENTATION_DIR / "isic_masks_manifest.csv"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.images_dir = args.images_dir.expanduser().absolute()
    args.masks_dir = args.masks_dir.expanduser().absolute()
    args.output_manifest = args.output_manifest.expanduser().absolute()
    args.checkpoint = args.checkpoint.expanduser().absolute()

    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest exists; pass --overwrite: {args.output_manifest}")

    _log(f"[1/5] Reading ISIC manifest: {args.manifest}")
    source = pd.read_csv(args.manifest)
    require_columns(source, ["image_id", "split"], name="ISIC split manifest")
    require_unique(source, "image_id", name="ISIC split manifest")
    image_ids = source["image_id"].astype(str).tolist()
    expected_ids = set(image_ids)
    _log(f"      Selected images: {len(image_ids):,}")

    if not args.images_dir.is_dir():
        raise NotADirectoryError(f"ISIC images directory not found: {args.images_dir}")
    _log(f"[2/5] Indexing mounted ISIC directory once: {args.images_dir}")
    available = _index_images(args.images_dir)
    matched = sum(image_id in available for image_id in image_ids)
    _log(f"      Indexed files: {len(available):,}; matched manifest IDs: {matched:,}")
    if matched == 0:
        raise FileNotFoundError(
            "No manifest image IDs were found in the flat images directory. "
            f"Check --images-dir (current value: {args.images_dir}); the files may be "
            "inside an images/ subdirectory."
        )
    if matched < len(image_ids):
        _log(f"      Warning: {len(image_ids) - matched:,} manifest images are missing")

    args.masks_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _log("[3/5] Overwrite enabled: skipping existing-mask scan")
    else:
        _log(f"[3/5] Checking existing masks with one directory listing: {args.masks_dir}")
        existing = _find_existing_masks(args.masks_dir, expected_ids)
        if existing:
            raise FileExistsError(
                f"{len(existing)} target masks already exist; pass --overwrite to replace them"
            )

    _log(f"[4/5] Loading configuration and checkpoint: {args.checkpoint}")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_mask_rcnn(num_classes=int(config["num_classes"]), pretrained=False)
    load_model_checkpoint(model, args.checkpoint, device=device)
    model.to(device).eval()
    thresholds = config["evaluation"]
    _log(f"      Device: {device}")
    if device.type == "cuda":
        _log(f"      GPU: {torch.cuda.get_device_name(device)}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[5/5] Segmenting {len(image_ids):,} ISIC images")
    with args.output_manifest.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for image_id in tqdm(
            image_ids,
            desc="Segmenting ISIC",
            unit="image",
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            image_path = available.get(image_id)
            mask_path = args.masks_dir / f"{image_id}.png"
            record = {
                "image_id": image_id,
                "image_path": str(image_path) if image_path else "",
                "mask_path": "",
                "score": "",
                "num_detections": 0,
                "status": "missing_image",
                "error": "",
            }
            if image_path is not None:
                try:
                    with Image.open(image_path) as handle:
                        image = handle.convert("RGB")
                    tensor = vision_functional.to_tensor(image).to(device)
                    with torch.inference_mode():
                        output = model([tensor])[0]
                    mask, score, count = best_binary_mask(
                        output,
                        (image.height, image.width),
                        score_threshold=float(thresholds["score_threshold"]),
                        mask_threshold=float(thresholds["mask_threshold"]),
                    )
                    mask_image = Image.fromarray(
                        mask.detach().cpu().numpy().astype("uint8") * 255, mode="L"
                    )
                    mask_image.save(mask_path)
                    record.update(
                        mask_path=str(mask_path),
                        score="" if score is None else score,
                        num_detections=count,
                        status="segmented" if count else "no_detection",
                    )
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            writer.writerow(record)
            output_handle.flush()
    _log(f"Wrote segmentation contract to {args.output_manifest}")


if __name__ == "__main__":
    main()
