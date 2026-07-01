"""
utils.py
Shared helper utilities used across the project.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for servers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "plant_disease", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed fixed to {seed}")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = True,
) -> None:
    """Save a PyTorch checkpoint dict to *filepath*."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        logger.info(f"✓ Best checkpoint saved → {filepath}")


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load a checkpoint from *filepath*."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device)
    logger.info(f"Loaded checkpoint from '{filepath}'")
    return checkpoint


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


class AverageMeter:
    """Tracks a running average (loss, accuracy, …)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """
    Plot loss and accuracy curves from a *history* dict.

    Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="royalblue")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {save_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def save_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
) -> Dict:
    """Persist sklearn classification report as JSON and return it."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved → {save_path}")
    return report


# ---------------------------------------------------------------------------
# Model size utility
# ---------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: torch.nn.Module) -> float:
    """Approximate model size in MB (float32 weights)."""
    total, _ = count_parameters(model)
    return total * 4 / (1024 ** 2)
