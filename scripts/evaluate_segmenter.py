#!/usr/bin/env python3
"""Evaluate a fixed Mask R-CNN checkpoint on HAM10000 test only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from scd_ml.data.schemas import assert_disjoint_groups
from scd_ml.paths import HAM10000_DIR, MODELS_DIR, SEGMENTATION_EVALUATION_DIR, SPLITS_DIR
from scd_ml.segmentation.dataset import HAM10000SegmentationDataset, collate_detection_batch
from scd_ml.segmentation.metrics import evaluate_segmenter
from scd_ml.segmentation.model import build_mask_rcnn, load_model_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "ham10000_segmentation.csv")
    parser.add_argument("--ham-root", type=Path, default=HAM10000_DIR)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=MODELS_DIR / "segmentation" / "mask_rcnn_best.pt"
    )
    parser.add_argument("--output-dir", type=Path, default=SEGMENTATION_EVALUATION_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary_path = args.output_dir / "ham10000_test_summary.csv"
    per_image_path = args.output_dir / "ham10000_test_per_image.csv"
    if not args.overwrite and (summary_path.exists() or per_image_path.exists()):
        raise FileExistsError("Evaluation output exists; pass --overwrite")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = pd.read_csv(args.manifest)
    assert_disjoint_groups(manifest)
    dataset = HAM10000SegmentationDataset(args.ham_root, manifest, split="test")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_detection_batch,
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_mask_rcnn(num_classes=int(config["num_classes"]), pretrained=False)
    load_model_checkpoint(model, args.checkpoint, device=device)
    model.to(device)
    thresholds = config["evaluation"]
    summary, per_image = evaluate_segmenter(
        model,
        loader,
        device,
        score_threshold=float(thresholds["score_threshold"]),
        mask_threshold=float(thresholds["mask_threshold"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    per_image.to_csv(per_image_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
