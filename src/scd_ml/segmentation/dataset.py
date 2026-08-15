"""HAM10000 dataset backed by the lesion-grouped split manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional


class DetectionTransform:
    def __init__(self, *, training: bool) -> None:
        self.training = training

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        tensor = vision_functional.to_tensor(image)
        if self.training and torch.rand(1).item() < 0.5:
            tensor = vision_functional.hflip(tensor)
            width = tensor.shape[-1]
            if len(target["boxes"]):
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            target["masks"] = target["masks"].flip(-1)
        return tensor, target


class HAM10000SegmentationDataset(torch.utils.data.Dataset):
    """Load image/mask pairs selected by a HAM10000 split manifest."""

    def __init__(
        self,
        root: str | Path,
        manifest: pd.DataFrame,
        *,
        split: str,
        training: bool = False,
    ) -> None:
        self.root = Path(root)
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"HAM10000 manifest contains no rows for split={split!r}")
        self.transform = DetectionTransform(training=training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        row = self.rows.iloc[index]
        image_path = self.root / "images" / row["image_filename"]
        mask_path = self.root / "masks" / row["mask_filename"]

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask_array = np.asarray(handle.convert("L")) > 127

        if mask_array.shape != (image.height, image.width):
            raise ValueError(
                f"Image/mask dimensions differ for {row['image_id']}: "
                f"image={(image.height, image.width)}, mask={mask_array.shape}"
            )

        if mask_array.any():
            ys, xs = np.where(mask_array)
            boxes = torch.tensor(
                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32
            )
            masks = torch.as_tensor(mask_array[None, ...], dtype=torch.uint8)
            labels = torch.ones((1,), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "source_id": str(row["image_id"]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return self.transform(image, target)


def collate_detection_batch(batch: list[tuple[torch.Tensor, dict]]) -> tuple[tuple, tuple]:
    return tuple(zip(*batch))
