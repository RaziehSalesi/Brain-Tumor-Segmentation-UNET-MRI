import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms() -> A.Compose:
    """Data augmentations applied on-the-fly during training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    """Tensor conversion without geometric augmentation for validation/test."""
    return A.Compose([ToTensorV2()])