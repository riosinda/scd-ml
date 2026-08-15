#!/usr/bin/env python3
"""Train Mask R-CNN on the grouped HAM10000 segmentation manifest."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scd_ml.data.schemas import assert_disjoint_groups, require_columns
from scd_ml.paths import (
    HAM10000_DIR,
    MODELS_DIR,
    SEGMENTATION_EVALUATION_DIR,
    SEGMENTATION_TRAINING_DIR,
    SPLITS_DIR,
)
from scd_ml.segmentation.dataset import HAM10000SegmentationDataset, collate_detection_batch
from scd_ml.segmentation.metrics import evaluate_segmenter
from scd_ml.segmentation.model import build_mask_rcnn
from scd_ml.segmentation.training import train_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "ham10000_segmentation.csv")
    parser.add_argument("--ham-root", type=Path, default=HAM10000_DIR)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=MODELS_DIR / "segmentation" / "mask_rcnn_best.pt"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _loader(dataset, config: dict, *, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_detection_batch,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )


def main() -> None:
    args = parse_args()
    planned_outputs = [
        args.checkpoint,
        SEGMENTATION_TRAINING_DIR / "training_history.csv",
        SEGMENTATION_EVALUATION_DIR / "ham10000_test_summary.csv",
        SEGMENTATION_EVALUATION_DIR / "ham10000_test_per_image.csv",
    ]
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{len(existing_outputs)} training outputs exist; pass --overwrite. "
            f"First: {existing_outputs[0]}"
        )
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = pd.read_csv(args.manifest)
    require_columns(
        manifest,
        ["image_id", "lesion_id", "group_id", "split", "image_filename", "mask_filename"],
        name="HAM10000 manifest",
    )
    assert_disjoint_groups(manifest)
    train_data = HAM10000SegmentationDataset(args.ham_root, manifest, split="train", training=True)
    val_data = HAM10000SegmentationDataset(args.ham_root, manifest, split="val")
    test_data = HAM10000SegmentationDataset(args.ham_root, manifest, split="test")
    train_loader = _loader(train_data, config, shuffle=True)
    val_loader = _loader(val_data, config, shuffle=False)
    test_loader = _loader(test_data, config, shuffle=False)

    device = _device(args.device)
    print(f"Device: {device}; train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    model = build_mask_rcnn(
        num_classes=int(config["num_classes"]), pretrained=not args.no_pretrained
    ).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["lr_step_size"]),
        gamma=float(config["lr_gamma"]),
    )
    evaluation = config["evaluation"]
    stopping = config["early_stopping"]
    if stopping.get("monitor") != "val_dice" or stopping.get("mode") != "max":
        raise ValueError("Segmentation early stopping must maximize val_dice")
    history, _ = train_model(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        epochs=int(config["epochs"]),
        checkpoint_path=args.checkpoint,
        patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]),
        score_threshold=float(evaluation["score_threshold"]),
        mask_threshold=float(evaluation["mask_threshold"]),
    )

    SEGMENTATION_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(SEGMENTATION_TRAINING_DIR / "training_history.csv", index=False)
    summary, per_image = evaluate_segmenter(
        model,
        test_loader,
        device,
        score_threshold=float(evaluation["score_threshold"]),
        mask_threshold=float(evaluation["mask_threshold"]),
    )
    summary.to_csv(SEGMENTATION_EVALUATION_DIR / "ham10000_test_summary.csv", index=False)
    per_image.to_csv(SEGMENTATION_EVALUATION_DIR / "ham10000_test_per_image.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
