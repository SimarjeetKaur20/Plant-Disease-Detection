"""
train_teacher.py
Full training loop for the Teacher model.

Usage:
    python train_teacher.py

Training strategy:
    Phase 1 – warm-up  (epochs 1‥WARMUP_EPOCHS):  backbone frozen, only head trained.
    Phase 2 – fine-tune (remaining epochs):         full network trained with lower LR.
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from config import (
    TEACHER_EPOCHS, TEACHER_BATCH_SIZE, TEACHER_LR,
    TEACHER_WEIGHT_DECAY, TEACHER_SCHEDULER, TEACHER_SAVE_PATH,
    NUM_WORKERS, RESULT_DIR, RANDOM_SEED, CLASS_NAMES,
)
from data_loader   import get_dataloaders
from teacher_model import build_teacher
from utils import (
    get_device, set_seed, get_logger,
    AverageMeter, compute_accuracy,
    save_checkpoint, plot_training_curves,
)

logger = get_logger(__name__)

WARMUP_EPOCHS = 999        # backbone frozen during these epochs


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    for step, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        acc = compute_accuracy(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc,          images.size(0))

        if (step + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch:03d} | Step {step+1:04d}/{len(loader):04d} | "
                f"Loss {loss_meter.avg:.4f} | Acc {acc_meter.avg:.4f}"
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter  = AverageMeter("val_acc")

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss    = criterion(outputs, targets)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(compute_accuracy(outputs, targets), images.size(0))

    return loss_meter.avg, acc_meter.avg


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_teacher():
    set_seed(RANDOM_SEED)
    device = get_device()

    # Data
    logger.info("Loading data …")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=TEACHER_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Model
    model = build_teacher(device, freeze_backbone=True)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser  (only head parameters while backbone is frozen)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TEACHER_LR,
        weight_decay=TEACHER_WEIGHT_DECAY,
    )

    # Scheduler
    if TEACHER_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS, eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # History
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(f"Starting teacher training for {TEACHER_EPOCHS} epochs …")
    t0 = time.time()

    for epoch in range(1, TEACHER_EPOCHS + 1):

        # Phase transition: unfreeze backbone after warm-up
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_backbone()
            # Re-create optimiser with smaller LR for backbone
            optimizer = optim.AdamW(
                model.parameters(),
                lr=TEACHER_LR * 0.1,
                weight_decay=TEACHER_WEIGHT_DECAY,
            )
            if TEACHER_SCHEDULER == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=TEACHER_EPOCHS - WARMUP_EPOCHS,
                    eta_min=1e-7,
                )
            logger.info("── Phase 2: full fine-tuning started ──")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = (time.time() - t0) / 60
        logger.info(
            f"[Epoch {epoch:03d}/{TEACHER_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Elapsed: {elapsed:.1f} min"
        )

        # Save best
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "class_names":      CLASS_NAMES,
                },
                TEACHER_SAVE_PATH,
                is_best=True,
            )

    total_time = (time.time() - t0) / 60
    logger.info(f"Training complete. Best val acc: {best_val_acc:.4f} | Total: {total_time:.1f} min")

    # Persist history
    history_path = os.path.join(RESULT_DIR, "teacher_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_training_curves(
        history,
        save_path=os.path.join(RESULT_DIR, "teacher_training_curves.png"),
        title="Teacher (ResNet-18) Training Curves",
    )

    return best_val_acc


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_teacher()
