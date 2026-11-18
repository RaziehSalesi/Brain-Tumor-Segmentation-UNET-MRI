from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import dice_coef, compute_iou, pixel_accuracy
from .early_stopping import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss, total_acc, total_iou, total_dice = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        optimizer.zero_grad()
        masks_pred = model(images)
        loss = loss_fn(masks_pred, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        total_acc += pixel_accuracy(masks_pred, masks)
        total_iou += compute_iou(masks, masks_pred)
        total_dice += dice_coef(masks, masks_pred)

    return {
        "loss": total_loss / num_batches,
        "acc": total_acc / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss, total_acc, total_iou, total_dice = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        masks_pred = model(images)
        loss = loss_fn(masks_pred, masks)

        total_loss += loss.detach().item()
        total_acc += pixel_accuracy(masks_pred, masks)
        total_iou += compute_iou(masks, masks_pred)
        total_dice += dice_coef(masks, masks_pred)

    return {
        "loss": total_loss / num_batches,
        "acc": total_acc / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    patience: int = 10,
    checkpoint_path: str | Path = "checkpoints/best_unet.pt",
    metrics_csv_path: str | Path | None = "results/metrics.csv",
) -> dict[str, list[float]]:
    """Run full training loop with EarlyStopping and optional CSV logging."""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
    }

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        early_stopping(val_metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if metrics_csv_path is not None:
        Path(metrics_csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(metrics_csv_path, index_label="epoch")

    return history