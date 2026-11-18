import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler


def discover_image_mask_pairs(dataset_dir: str | Path) -> pd.DataFrame:
    """Recursively find and match MRI slices and masks, extracting patient IDs."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    all_files = list(dataset_path.rglob("*.tif"))
    image_dict = {}
    mask_dict = {}

    for file in all_files:
        if file.stem.endswith("_mask"):
            image_name = file.stem[:-5]
            mask_dict[(file.parent, image_name)] = str(file)
        else:
            image_dict[(file.parent, file.stem)] = str(file)

    common_keys = sorted(
        set(image_dict.keys()) & set(mask_dict.keys()),
        key=lambda k: (k[0].name, int(k[1].rsplit("_", 1)[1])),
    )

    records = []
    for key in common_keys:
        folder, filename = key
        records.append(
            {
                "ID": filename,
                "patient_id": folder.name,
                "image_path": image_dict[key],
                "mask_path": mask_dict[key],
            }
        )

    return pd.DataFrame(records)


def preprocess_images_and_masks(
    df: pd.DataFrame, img_size: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize, normalize masks to [0, 1], and apply per-channel MinMaxScaler to images."""
    img_nor = []
    mask_nor = []
    valid_patients = []

    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"])
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue

        # Bilinear resize for MRI image, nearest-neighbor for binary mask
        img_resized = cv2.resize(img, img_size)
        mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        # Normalize mask to {0.0, 1.0}
        mask_float = (mask_resized.astype(np.float32)) / 255.0

        # Normalize each channel of the MRI image independently
        img_reshaped = img_resized.reshape(-1, img_resized.shape[-1])
        scaler = MinMaxScaler()
        img_scaled = scaler.fit_transform(img_reshaped).reshape(img_resized.shape).astype(np.float32)

        img_nor.append(img_scaled)
        mask_nor.append(mask_float)
        valid_patients.append(row["patient_id"])

    return np.array(img_nor), np.array(mask_nor), np.array(valid_patients)


def load_and_split_data(
    dataset_dir: str | Path,
    img_size: tuple[int, int] = (128, 128),
    test_size: float = 0.2,
    val_test_split: float = 0.5,
    random_state: int = 42,
):
    """Load dataset and perform leak-free patient-level Train/Val/Test split."""
    df = discover_image_mask_pairs(dataset_dir)
    img_arr, mask_arr, patient_arr = preprocess_images_and_masks(df, img_size=img_size)

    # Split 80% patients Train, 20% Val+Test
    gss_train = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(img_arr, mask_arr, groups=patient_arr))

    X_train, y_train = img_arr[train_idx], mask_arr[train_idx]
    X_temp, y_temp = img_arr[temp_idx], mask_arr[temp_idx]
    patients_temp = patient_arr[temp_idx]

    # Split remaining 20% patients equally into 10% Val and 10% Test
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=val_test_split, random_state=random_state)
    val_idx, test_idx = next(gss_val_test.split(X_temp, y_temp, groups=patients_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    # Guarantee zero patient overlap
    assert set(patient_arr[train_idx]).isdisjoint(set(patients_temp[val_idx]))
    assert set(patient_arr[train_idx]).isdisjoint(set(patients_temp[test_idx]))
    assert set(patients_temp[val_idx]).isdisjoint(set(patients_temp[test_idx]))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), df