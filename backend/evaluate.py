"""
evaluate.py
Comprehensive evaluation of Teacher and/or Student models on the test set.

Metrics reported:
    • Top-1 and Top-5 accuracy
    • Per-class precision, recall, F1 (saved as JSON)
    • Confusion matrix heatmap  (PNG)
    • Training-curve comparison (PNG) if both histories exist
    • Inference latency and model size comparison

Usage:
    python evaluate.py                      # evaluates both models
    python evaluate.py --model teacher
    python evaluate.py --model student
"""

import os
import json
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from config import (
    TEACHER_SAVE_PATH, STUDENT_SAVE_PATH,
    CLASS_NAMES, NUM_CLASSES, RESULT_DIR,
    NUM_WORKERS,
)
from data_loader    import get_dataloaders
from teacher_model  import build_teacher
from student_model  import build_student
from utils import (
    get_device, get_logger,
    plot_confusion_matrix,
    save_classification_report,
    count_parameters, model_size_mb,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
    top_k:   int = 5,
) -> dict:
    """
    Run inference on *loader* and return a metrics dict:
        top1_acc, top5_acc, avg_loss,
        all_preds (List[int]), all_targets (List[int]),
        inference_time_ms (per image)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds:   List[int] = []
    all_targets: List[int] = []
    total_loss   = 0.0
    top1_correct = 0
    top5_correct = 0
    total        = 0
    total_time   = 0.0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        t_start = time.perf_counter()
        outputs = model(images)
        t_end   = time.perf_counter()
        total_time += (t_end - t_start)

        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)

        # Top-1
        preds = outputs.argmax(dim=1)
        top1_correct += (preds == targets).sum().item()

        # Top-5
        k = min(top_k, NUM_CLASSES)
        _, top5_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
        top5_correct += (
            top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds))
            .any(dim=1)
            .sum()
            .item()
        )

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        total += images.size(0)

    avg_loss        = total_loss / total
    top1_acc        = top1_correct / total
    top5_acc        = top5_correct / total
    latency_ms      = (total_time / total) * 1000   # per image

    return {
        "top1_acc":          top1_acc,
        "top5_acc":          top5_acc,
        "avg_loss":          avg_loss,
        "all_preds":         all_preds,
        "all_targets":       all_targets,
        "inference_time_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline for one model
# ---------------------------------------------------------------------------
def run_evaluation(model_type: str, device: torch.device, test_loader) -> dict:
    """
    Load the specified model, run evaluation, save all artifacts, and return
    a summary dict.
    """
    assert model_type in ("teacher", "student")

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_type.upper()} model …")
    logger.info(f"{'='*60}")

    # Load model
    if model_type == "teacher":
        model = build_teacher(device, checkpoint_path=TEACHER_SAVE_PATH)
        tag   = "teacher"
    else:
        model = build_student(device, checkpoint_path=STUDENT_SAVE_PATH)
        tag   = "student"

    # Stats
    total_params, _ = count_parameters(model)
    size_mb         = model_size_mb(model)

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    logger.info(f"  Top-1 Accuracy : {results['top1_acc']*100:.2f} %")
    logger.info(f"  Top-5 Accuracy : {results['top5_acc']*100:.2f} %")
    logger.info(f"  Avg CE Loss    : {results['avg_loss']:.4f}")
    logger.info(f"  Latency        : {results['inference_time_ms']:.2f} ms / image")
    logger.info(f"  Parameters     : {total_params:,}  (~{size_mb:.1f} MB)")

    # Confusion matrix
    plot_confusion_matrix(
        y_true      = results["all_targets"],
        y_pred      = results["all_preds"],
        class_names = CLASS_NAMES,
        save_path   = os.path.join(RESULT_DIR, f"{tag}_confusion_matrix.png"),
        title       = f"{tag.capitalize()} – Confusion Matrix (test set)",
        normalize   = True,
    )

    # Classification report
    report = save_classification_report(
        y_true      = results["all_targets"],
        y_pred      = results["all_preds"],
        class_names = CLASS_NAMES,
        save_path   = os.path.join(RESULT_DIR, f"{tag}_classification_report.json"),
    )

    # Per-class accuracy from report
    logger.info("\n  Per-class F1-scores:")
    for cls in CLASS_NAMES:
        f1 = report.get(cls, {}).get("f1-score", 0.0)
        logger.info(f"    {cls:<55s}: {f1:.4f}")

    summary = {
        "model":             tag,
        "top1_acc":          results["top1_acc"],
        "top5_acc":          results["top5_acc"],
        "avg_loss":          results["avg_loss"],
        "latency_ms":        results["inference_time_ms"],
        "total_params":      total_params,
        "size_mb":           size_mb,
        "macro_f1":          report.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_f1":       report.get("weighted avg", {}).get("f1-score", 0.0),
    }
    return summary


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(teacher_summary: dict, student_summary: dict):
    logger.info("\n" + "="*70)
    logger.info("TEACHER vs STUDENT COMPARISON")
    logger.info("="*70)
    metrics = [
        ("Top-1 Accuracy",  "top1_acc",    "{:.4f}"),
        ("Top-5 Accuracy",  "top5_acc",    "{:.4f}"),
        ("Macro F1",        "macro_f1",    "{:.4f}"),
        ("Latency (ms/img)","latency_ms",  "{:.2f}"),
        ("Parameters",      "total_params","{:,}"),
        ("Model Size (MB)", "size_mb",     "{:.1f}"),
    ]
    fmt = "{:<22s} {:>15s} {:>15s}"
    logger.info(fmt.format("Metric", "Teacher", "Student"))
    logger.info("-"*55)
    for label, key, val_fmt in metrics:
        t_val = val_fmt.format(teacher_summary[key])
        s_val = val_fmt.format(student_summary[key])
        logger.info(fmt.format(label, t_val, s_val))

    compression = teacher_summary["total_params"] / max(student_summary["total_params"], 1)
    speedup     = teacher_summary["latency_ms"]   / max(student_summary["latency_ms"],    1e-9)
    logger.info(f"\n  Compression ratio : {compression:.1f}×")
    logger.info(f"  Speedup           : {speedup:.1f}×")
    logger.info("="*70)

    # Save combined summary
    combined = {
        "teacher": teacher_summary,
        "student": student_summary,
        "compression_ratio": compression,
        "speedup":           speedup,
    }
    out_path = os.path.join(RESULT_DIR, "evaluation_summary.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Summary saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate plant disease models")
    parser.add_argument(
        "--model",
        choices=["teacher", "student", "both"],
        default="both",
        help="Which model to evaluate (default: both)",
    )
    args = parser.parse_args()

    device = get_device()
    _, _, test_loader = get_dataloaders(num_workers=NUM_WORKERS)

    teacher_summary = student_summary = None

    if args.model in ("teacher", "both"):
        teacher_summary = run_evaluation("teacher", device, test_loader)

    if args.model in ("student", "both"):
        student_summary = run_evaluation("student", device, test_loader)

    if teacher_summary and student_summary:
        print_comparison(teacher_summary, student_summary)


if __name__ == "__main__":
    main()
