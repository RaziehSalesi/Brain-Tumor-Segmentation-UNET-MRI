import numpy as np
import torch


def dice_coef(mask_true: torch.Tensor, mask_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute batch-averaged Dice Similarity Coefficient."""
    mask_pred = (mask_pred > threshold).float()
    dice_list = []
    for true_m, pred_m in zip(mask_true, mask_pred):
        intersection = torch.sum(true_m * pred_m, dim=[1, 2])
        union = torch.sum(true_m, dim=[1, 2]) + torch.sum(pred_m, dim=[1, 2])
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_list.append(dice.cpu().item())
    return float(np.mean(dice_list))


def compute_iou(mask_true: torch.Tensor, mask_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute batch-averaged Intersection over Union (Mean IoU)."""
    mask_pred = (mask_pred > threshold).float()
    mask_true_flat = mask_true.view(mask_true.size(0), -1)
    mask_pred_flat = mask_pred.view(mask_pred.size(0), -1)

    intersection = torch.sum(mask_true_flat * mask_pred_flat, dim=1)
    union = torch.sum(mask_true_flat, dim=1) + torch.sum(mask_pred_flat, dim=1) - intersection
    iou = (intersection + 1e-10) / (union + 1e-10)
    return float(iou.mean().cpu().item())


def pixel_accuracy(mask_pred: torch.Tensor, mask_true: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute pixel-wise binary segmentation accuracy."""
    with torch.no_grad():
        batch_size = mask_true.size(0)
        mask_pred_flat = (mask_pred.view(batch_size, -1) > threshold).float()
        mask_true_flat = mask_true.view(batch_size, -1)
        correct = (mask_pred_flat == mask_true_flat).float().sum(dim=1)
        acc = correct / mask_true_flat.size(1)
        return float(acc.mean().item())