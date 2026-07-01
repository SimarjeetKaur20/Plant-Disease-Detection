"""
data_loader.py
Builds train / val / test DataLoaders from the PlantVillage folder structure.

Directory layout expected:
    dataset/PlantVillage/
        <ClassName>/
            *.JPG  (or .jpg / .png)

The module splits images per-class (stratified) so class proportions are
preserved across splits.
"""

import os
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    DATA_DIR, CLASS_NAMES, IMAGE_SIZE, MEAN, STD,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    TEACHER_BATCH_SIZE, NUM_WORKERS,
)
from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------
def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.14), int(IMAGE_SIZE[1] * 1.14))),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ---------------------------------------------------------------------------
# Custom dataset
# ---------------------------------------------------------------------------
class PlantDiseaseDataset(Dataset):
    """
    Loads (image_path, label) pairs and applies *transform* on-the-fly.
    Corrupt / unreadable images are skipped with a warning.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path  = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning(f"Cannot read {path}: {exc}. Returning blank image.")
            image = Image.new("RGB", IMAGE_SIZE, color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def _collect_samples(data_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """Walk *data_dir* and return (paths, labels) for all recognised classes."""
    paths, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    logger.info(f"Collected {len(paths)} images across {len(class_names)} classes.")
    return paths, labels


def build_datasets(
    data_dir: str = DATA_DIR,
    class_names: List[str] = CLASS_NAMES,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio:   float = VAL_SPLIT,
    seed:        int   = RANDOM_SEED,
) -> Tuple[PlantDiseaseDataset, PlantDiseaseDataset, PlantDiseaseDataset]:
    """
    Returns (train_dataset, val_dataset, test_dataset) with appropriate
    transforms applied.  Split is stratified by class label.
    """
    paths, labels = _collect_samples(data_dir, class_names)

    # First split: train vs (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test  (from the leftover portion)
    val_frac = val_ratio / (val_ratio + test_ratio)
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_temp, labels_temp,
        test_size=(1.0 - val_frac),
        stratify=labels_temp,
        random_state=seed,
    )

    logger.info(
        f"Split → Train: {len(paths_train)}, "
        f"Val: {len(paths_val)}, "
        f"Test: {len(paths_test)}"
    )

    train_ds = PlantDiseaseDataset(paths_train, labels_train, get_train_transforms())
    val_ds   = PlantDiseaseDataset(paths_val,   labels_val,   get_eval_transforms())
    test_ds  = PlantDiseaseDataset(paths_test,  labels_test,  get_eval_transforms())
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    batch_size:  int = TEACHER_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = build_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")
    print(f"Label range : {labels.min().item()} – {labels.max().item()}")
