from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when validation loss stops improving and save best model weights."""

    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0,
        path: str | Path = "checkpoints/best_unet.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = Path(path)
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss