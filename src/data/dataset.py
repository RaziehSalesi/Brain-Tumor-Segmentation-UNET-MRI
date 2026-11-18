import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import get_train_transforms, get_val_transforms


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for 2D MRI slices and binary tumor masks."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image.float(), mask.float()


def create_dataloaders(
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    train_dataset = BrainTumorDataset(*train_data, transform=get_train_transforms())
    val_dataset = BrainTumorDataset(*val_data, transform=get_val_transforms())
    test_dataset = BrainTumorDataset(*test_data, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader