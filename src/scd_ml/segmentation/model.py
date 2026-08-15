"""Mask R-CNN construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_mask_rcnn(*, num_classes: int = 2, pretrained: bool = True):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = (
        torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        if pretrained
        else None
    )
    kwargs: dict[str, Any] = {"weights": weights}
    if not pretrained:
        kwargs["weights_backbone"] = None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(**kwargs)

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def load_model_checkpoint(model, checkpoint_path: str | Path, *, device):
    import torch

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state = (
        checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )
    model.load_state_dict(state)
    return checkpoint
