"""Deprecated compatibility import for the corrected evaluator."""

from scd_ml.segmentation.metrics import evaluate_segmenter


def evaluate(model, data_loader, device, threshold=0.5):
    return evaluate_segmenter(
        model,
        data_loader,
        device,
        score_threshold=0.5,
        mask_threshold=threshold,
    )
