import torch.nn as nn
import torch.optim as optim

from src.config import (
    DATA_DIR, IMG_SIZE, IN_CHANNELS, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    CHECKPOINT_PATH, METRICS_CSV_PATH, DEVICE
)
from src.data import load_and_split_data, create_dataloaders
from src.models import UNetModel
from src.training import fit
from src.evaluation import evaluate_model


def main():
    print("[-] 1. Loading and splitting dataset (patient-level to prevent data leakage)...")
    train_data, val_data, test_data, df = load_and_split_data(DATA_DIR, img_size=IMG_SIZE)
    print(f"    - Train samples: {len(train_data[0])}")
    print(f"    - Validation samples: {len(val_data[0])}")
    print(f"    - Test samples: {len(test_data[0])}")

    print("\n[-] 2. Creating PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )

    print("\n[-] 3. Initializing U-Net model, loss function, and optimizer...")
    model = UNetModel(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n[-] 4. Starting training loop with Early Stopping...")
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=CHECKPOINT_PATH,
        metrics_csv_path=METRICS_CSV_PATH,
    )
    print(f"[+] Training completed. Best model saved to {CHECKPOINT_PATH}")

    print("\n[-] 5. Evaluating best model on the held-out test set...")
    test_results = evaluate_model(test_loader, model, loss_fn, DEVICE)
    
    print("\n" + "="*30)
    print("[+] FINAL TEST RESULTS:")
    print("="*30)
    for metric_name, value in test_results.items():
        print(f"  - {metric_name}: {value:.4f}")
    print("="*30)


if __name__ == "__main__":
    main()