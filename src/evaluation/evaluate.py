import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.metrics import dice_coef, compute_iou, pixel_accuracy


@torch.no_grad()
def evaluate_model(
    test_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate trained U-Net model on the held-out patient test set."""
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    model.eval()
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        batch_size = masks.size(0)

        masks_pred = model(images)
        loss = loss_fn(masks_pred, masks)

        total_loss += loss.item() * batch_size
        total_accuracy += pixel_accuracy(masks_pred, masks) * batch_size
        total_iou += compute_iou(masks, masks_pred) * batch_size
        total_dice += dice_coef(masks, masks_pred) * batch_size
        total_samples += batch_size

    return {
        "test_accuracy": total_accuracy / total_samples,
        "test_loss": total_loss / total_samples,
        "test_iou": total_iou / total_samples,
        "test_dice": total_dice / total_samples,
    }