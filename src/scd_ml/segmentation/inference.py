"""Mask selection used by ISIC inference."""

from __future__ import annotations


def best_binary_mask(
    output,
    image_size: tuple[int, int],
    *,
    score_threshold: float,
    mask_threshold: float,
):
    """Return ``(mask, score, valid_detection_count)`` for one model output."""
    import torch
    import torch.nn.functional as functional

    scores = output.get("scores")
    masks = output.get("masks")
    if scores is None or masks is None:
        return torch.zeros(image_size, dtype=torch.bool), None, 0
    valid = torch.where(scores >= score_threshold)[0]
    if len(valid) == 0:
        return torch.zeros(image_size, dtype=torch.bool, device=scores.device), None, 0
    best = valid[torch.argmax(scores[valid])]
    probability = masks[best, 0]
    if tuple(probability.shape) != image_size:
        probability = functional.interpolate(
            probability[None, None], size=image_size, mode="bilinear", align_corners=False
        )[0, 0]
    return probability >= mask_threshold, float(scores[best].item()), int(len(valid))
