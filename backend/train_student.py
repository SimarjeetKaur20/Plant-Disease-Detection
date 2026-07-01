"""
train_student.py
Knowledge Distillation training loop for the Student model.

Loss:
    L = alpha * KL_divergence(soft_student || soft_teacher)
      + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

where soft logits are scaled by temperature T before softmax.

Usage:
    python train_student.py
    # or with a custom teacher checkpoint:
    # python train_student.py  (reads TEACHER_SAVE_PATH from config)
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from config import (
    STUDENT_EPOCHS, STUDENT_BATCH_SIZE, STUDENT_LR,
    STUDENT_WEIGHT_DECAY, STUDENT_SAVE_PATH,
    TEACHER_SAVE_PATH, KD_TEMPERATURE, KD_ALPHA,
    NUM_WORKERS, RESULT_DIR, RANDOM_SEED, CLASS_NAMES,
)
from data_loader    import get_dataloaders
from teacher_model  import build_teacher
from student_model  import build_student
from utils import (
    get_device, set_seed, get_logger,
    AverageMeter, compute_accuracy,
    save_checkpoint, plot_training_curves,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Knowledge-Distillation loss
# ---------------------------------------------------------------------------
class DistillationLoss(nn.Module):
    """
    Combined hard-label CE + soft-label KL distillation loss.

    Args:
        temperature: Softening factor T.  Higher → softer distribution.
        alpha:       Weight of the distillation (KL) term.
                     Hard-label CE gets weight (1 - alpha).
    """

    def __init__(self, temperature: float = KD_TEMPERATURE, alpha: float = KD_ALPHA):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets:        torch.Tensor,
    ) -> torch.Tensor:
        # Hard-label loss
        ce_loss = self.ce(student_logits, targets)

        # Soft-label KL loss
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits    / self.T, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)

        return self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    student:   nn.Module,
    teacher:   nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: DistillationLoss,
    optimizer: optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, float]:
    student.train()
    teacher.eval()

    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc")

    for step, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()
        student_logits = student(images)
        loss = criterion(student_logits, teacher_logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()

        acc = compute_accuracy(student_logits, targets)
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
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
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
# Main
# ---------------------------------------------------------------------------
def train_student():
    set_seed(RANDOM_SEED)
    device = get_device()

    # Data
    logger.info("Loading data …")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=STUDENT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Teacher  (frozen – inference only)
    logger.info("Loading teacher model …")
    teacher = build_teacher(device, checkpoint_path=TEACHER_SAVE_PATH)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student
    student   = build_student(device)
    criterion = DistillationLoss(temperature=KD_TEMPERATURE, alpha=KD_ALPHA)
    optimizer = optim.AdamW(
        student.parameters(),
        lr=STUDENT_LR,
        weight_decay=STUDENT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=STUDENT_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(
        f"KD params → Temperature: {KD_TEMPERATURE}, Alpha: {KD_ALPHA}"
    )
    logger.info(f"Starting student training for {STUDENT_EPOCHS} epochs …")
    t0 = time.time()

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = (time.time() - t0) / 60
        logger.info(
            f"[Epoch {epoch:03d}/{STUDENT_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Elapsed: {elapsed:.1f} min"
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch":            epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_acc":          val_acc,
                    "val_loss":         val_loss,
                    "class_names":      CLASS_NAMES,
                    "kd_temperature":   KD_TEMPERATURE,
                    "kd_alpha":         KD_ALPHA,
                },
                STUDENT_SAVE_PATH,
                is_best=True,
            )

    total_time = (time.time() - t0) / 60
    logger.info(
        f"Student training complete. "
        f"Best val acc: {best_val_acc:.4f} | Total: {total_time:.1f} min"
    )

    # Persist history
    history_path = os.path.join(RESULT_DIR, "student_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(
        history,
        save_path=os.path.join(RESULT_DIR, "student_training_curves.png"),
        title="Student (MobileNetV2) Knowledge-Distillation Curves",
    )

    return best_val_acc


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_student()
