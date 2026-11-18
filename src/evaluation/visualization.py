from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_with_mask(path_img: str, path_mask: str) -> None:
    """Display raw MRI slice, ground-truth binary mask, and red overlay."""
    img_org = cv2.imread(path_img)
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    mask_org = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)

    mask_3d = np.expand_dims(mask_org, axis=2)
    red_overlay = np.where(mask_3d == 255, [255, 0, 0], img_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("MRI Image")
    axes[1].imshow(mask_org, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(red_overlay)
    axes[2].set_title("Tumor Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = "results/figures/training_curves.png",
) -> None:
    """Plot 2x2 grid of Loss, Pixel Accuracy, Mean IoU, and Dice Coefficient."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    axes[0, 0].plot(history["train_loss"], "-.", label="Train Loss")
    axes[0, 0].plot(history["val_loss"], "-.", label="Validation Loss")
    axes[0, 0].set_title("Loss", fontsize=14)
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # 2. Accuracy
    axes[0, 1].plot(history["train_acc"], "-.", label="Train Accuracy")
    axes[0, 1].plot(history["val_acc"], "-.", label="Validation Accuracy")
    axes[0, 1].set_title("Pixel Accuracy", fontsize=14)
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. Mean IoU
    axes[1, 0].plot(history["train_iou"], "-.", label="Train IoU")
    axes[1, 0].plot(history["val_iou"], "-.", label="Validation IoU")
    axes[1, 0].set_title("Mean IoU", fontsize=14)
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 4. Dice Coefficient
    axes[1, 1].plot(history["train_dice"], "-.", label="Train Dice")
    axes[1, 1].plot(history["val_dice"], "-.", label="Validation Dice")
    axes[1, 1].set_title("Dice Coefficient", fontsize=14)
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    sns.despine()
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_predictions(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_samples: int = 5,
    threshold: float = 0.5,
    save_path: str | Path | None = "results/figures/evaluation_examples.png",
) -> None:
    """Visualize MRI inputs alongside ground-truth and predicted tumor masks."""
    model.eval()
    images, masks = next(iter(test_loader))
    images_dev = images.to(device)
    preds = (model(images_dev) > threshold).float().cpu()

    num_samples = min(num_samples, images.size(0))
    indices = np.random.choice(images.size(0), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        img_hwc = images[idx].numpy().transpose(1, 2, 0)
        # Convert BGR (OpenCV) back to RGB for display
        img_rgb = img_hwc[..., ::-1]
        true_mask = masks[idx].numpy()
        pred_mask = preds[idx, 0].numpy()

        axes[row, 0].imshow(np.clip(img_rgb, 0, 1))
        axes[row, 0].set_title("MRI Input")
        axes[row, 1].imshow(true_mask, cmap="gray")
        axes[row, 1].set_title("True Mask")
        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title("Predicted Mask")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()