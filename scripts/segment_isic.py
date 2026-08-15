#!/usr/bin/env python3
"""Apply the HAM10000 segmenter to every image in an ISIC split manifest."""

from __future__ import annotations

import argparse
import csv
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

    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest exists; pass --overwrite: {args.output_manifest}")
    source = pd.read_csv(args.manifest)
    require_columns(source, ["image_id", "split"], name="ISIC split manifest")
    require_unique(source, "image_id", name="ISIC split manifest")
    available = {
        path.stem: path
        for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    expected = [args.masks_dir / f"{image_id}.png" for image_id in source["image_id"]]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} target masks already exist; pass --overwrite to replace them"
        )

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

    args.masks_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for image_id in tqdm(source["image_id"].astype(str), desc="Segmenting ISIC"):
            image_path = available.get(image_id)
            mask_path = args.masks_dir / f"{image_id}.png"
            record = {
                "image_id": image_id,
                "image_path": str(image_path.resolve()) if image_path else "",
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
                    with torch.no_grad():
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
                        mask_path=str(mask_path.resolve()),
                        score="" if score is None else score,
                        num_detections=count,
                        status="segmented" if count else "no_detection",
                    )
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            writer.writerow(record)
            output_handle.flush()
    print(f"Wrote segmentation contract to {args.output_manifest}")


if __name__ == "__main__":
    main()
