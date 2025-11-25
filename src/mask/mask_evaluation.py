import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def evaluate(model, data_loader, device, threshold=0.5):
    model.eval()

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    accuracy_scores = []

    epsilon = 1e-6

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # --- Ground truth ---
                target_mask = targets[i]['masks'].to(device)
                if target_mask.shape[0] == 0:
                    continue  # skip empty-mask images (already filtered)
                target_mask = (target_mask.max(dim=0)[0] > 0.5).float()

                # --- Prediction ---
                if len(output['scores']) == 0:
                    pred_mask = torch.zeros_like(target_mask)
                else:
                    idx = torch.argmax(output['scores'])
                    pred_mask = output['masks'][idx]

                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0),
                        size=target_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    pred_mask = (pred_mask > threshold).float()

                # --- Flatten ---
                pred_flat = pred_mask.view(-1)
                target_flat = target_mask.view(-1)

                TP = (pred_flat * target_flat).sum().item()
                FP = (pred_flat * (1 - target_flat)).sum().item()
                FN = ((1 - pred_flat) * target_flat).sum().item()
                TN = ((1 - pred_flat) * (1 - target_flat)).sum().item()

                # --- METRICS ---
                iou = (TP + epsilon) / (TP + FP + FN + epsilon)
                dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
                precision = (TP + epsilon) / (TP + FP + epsilon)
                recall    = (TP + epsilon) / (TP + FN + epsilon)
                specificity = (TN + epsilon) / (TN + FP + epsilon)
                accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)

                # --- Append ---
                iou_scores.append(iou)
                dice_scores.append(dice)
                precision_scores.append(precision)
                recall_scores.append(recall)
                specificity_scores.append(specificity)
                accuracy_scores.append(accuracy)

    means_values = {
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),

        'mean_iou': np.mean(iou_scores),
        'std_iou': np.std(iou_scores),

        'mean_precision': np.mean(precision_scores),
        'std_precision': np.std(precision_scores),

        'mean_recall': np.mean(recall_scores),
        'std_recall': np.std(recall_scores),

        'mean_specificity': np.mean(specificity_scores),
        'std_specificity': np.std(specificity_scores),

        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
    }

    original = {
        'dice_scores': dice_scores,
        'iou_scores': iou_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'specificity_scores': specificity_scores,
        'accuracy_scores': accuracy_scores,
    }

    metrics = {}
    std = []
    for k, v in means_values.items():
        if 'std' in k.split('_')[0]:
            std.append(v)
        else:
            metrics[k] = v

    metrics_df = pd.DataFrame([metrics])
    metric_final = metrics_df.melt(var_name='metric', value_name='value')
    metric_final['std'] = std

    all_results = pd.DataFrame(original)
    all_results = all_results.melt(var_name="metric", value_name="value")
    
    return metric_final, all_results