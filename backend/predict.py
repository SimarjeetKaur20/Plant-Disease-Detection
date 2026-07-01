"""
predict.py
Single-image and batch inference interface for the Plant Disease Detection
system.  Supports both Teacher (high-accuracy) and Student (fast) models.

Usage examples:
    # Single image
    python predict.py --image path/to/leaf.jpg

    # Batch prediction on a folder
    python predict.py --folder path/to/images/ --model student

    # Return raw JSON output
    python predict.py --image leaf.jpg --json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Union, List, Optional, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from config import (
    CLASS_NAMES, NUM_CLASSES,
    TEACHER_SAVE_PATH, STUDENT_SAVE_PATH,
    IMAGE_SIZE, MEAN, STD,
)
from teacher_model  import build_teacher
from student_model  import build_student
from data_loader    import get_eval_transforms
from utils          import get_device, get_logger

logger = get_logger(__name__)

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------
class PlantDiseasePredictor:
    """
    High-level inference wrapper that loads a model once and exposes
    predict() / predict_batch() methods.

    Args:
        model_type:  "teacher" | "student"
        device:      torch.device (auto-detected if None)
        top_k:       number of top predictions to return
    """

    def __init__(
        self,
        model_type: str = "student",
        device:     Optional[torch.device] = None,
        top_k:      int = 5,
    ):
        assert model_type in ("teacher", "student"), \
            f"model_type must be 'teacher' or 'student', got '{model_type}'"

        self.model_type = model_type
        self.device     = device or get_device()
        self.top_k      = min(top_k, NUM_CLASSES)
        self.transform  = get_eval_transforms()

        logger.info(f"Loading {model_type} model for inference …")
        if model_type == "teacher":
            self.model = build_teacher(self.device, checkpoint_path=TEACHER_SAVE_PATH)
        else:
            self.model = build_student(self.device, checkpoint_path=STUDENT_SAVE_PATH)
        self.model.eval()
        logger.info("Model ready.")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load an image from disk, apply eval transforms, return (1, C, H, W) tensor."""
        image = Image.open(str(image_path)).convert("RGB")
        tensor = self.transform(image)          # (C, H, W)
        return tensor.unsqueeze(0)              # (1, C, H, W)

    @torch.no_grad()
    def _infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Run forward pass, return probability tensor (B, NUM_CLASSES)."""
        batch = batch.to(self.device)
        logits = self.model(batch)
        return F.softmax(logits, dim=1)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict the disease class for a single image.

        Returns a dict with:
            predicted_class   – name of the top-1 class
            predicted_label   – integer index of the top-1 class
            confidence        – probability of the top-1 class (0 – 1)
            top_k_predictions – list of {class, label, confidence} dicts
            is_healthy        – True if the predicted class contains 'healthy'
            plant_type        – inferred plant type from class name
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        tensor = self._preprocess(path)
        probs  = self._infer(tensor)[0]   # (NUM_CLASSES,)

        top_probs, top_indices = probs.topk(self.top_k)
        top_probs   = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        top_k_preds = [
            {
                "class":      CLASS_NAMES[idx],
                "label":      int(idx),
                "confidence": float(prob),
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        best = top_k_preds[0]
        return {
            "image_path":      str(path),
            "predicted_class": best["class"],
            "predicted_label": best["label"],
            "confidence":      best["confidence"],
            "is_healthy":      "healthy" in best["class"].lower(),
            "plant_type":      _extract_plant_type(best["class"]),
            "top_k_predictions": top_k_preds,
        }

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size:  int = 16,
    ) -> List[Dict]:
        """
        Predict for a list of image paths.  Images are processed in batches
        for efficiency.

        Returns a list of prediction dicts (same format as predict()).
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            tensors = []
            valid_paths = []
            for p in chunk:
                try:
                    tensors.append(self._preprocess(p))
                    valid_paths.append(p)
                except Exception as exc:
                    logger.warning(f"Skipping {p}: {exc}")

            if not tensors:
                continue

            batch  = torch.cat(tensors, dim=0)   # (B, C, H, W)
            probs  = self._infer(batch)           # (B, NUM_CLASSES)

            for j, path in enumerate(valid_paths):
                p_row = probs[j]
                top_probs, top_indices = p_row.topk(self.top_k)
                top_probs   = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

                top_k_preds = [
                    {
                        "class":      CLASS_NAMES[idx],
                        "label":      int(idx),
                        "confidence": float(prob),
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ]
                best = top_k_preds[0]
                results.append({
                    "image_path":      str(path),
                    "predicted_class": best["class"],
                    "predicted_label": best["label"],
                    "confidence":      best["confidence"],
                    "is_healthy":      "healthy" in best["class"].lower(),
                    "plant_type":      _extract_plant_type(best["class"]),
                    "top_k_predictions": top_k_preds,
                })

            logger.info(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

        return results

    def predict_folder(self, folder_path: Union[str, Path], batch_size: int = 16) -> List[Dict]:
        """Predict for all supported images inside *folder_path*."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        paths = [p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS]
        if not paths:
            raise ValueError(f"No supported images found in {folder_path}")

        logger.info(f"Found {len(paths)} images in {folder}")
        return self.predict_batch(paths, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _extract_plant_type(class_name: str) -> str:
    """Infer plant type from class name (e.g. 'Tomato_Early_blight' → 'Tomato')."""
    name = class_name.replace("__", "_").replace("___", "_")
    parts = name.split("_")
    return parts[0].capitalize() if parts else "Unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")

    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="student",
        help="Which model to use for inference (default: student)",
    )
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predictions to display")
    parser.add_argument("--json", action="store_true",
                        help="Print output as JSON")
    parser.add_argument("--save", type=str, default=None,
                        help="Save predictions to this JSON file")
    args = parser.parse_args()

    predictor = PlantDiseasePredictor(model_type=args.model, top_k=args.top_k)

    if args.image:
        result = predictor.predict(args.image)
        output = [result]
    else:
        output = predictor.predict_folder(args.folder)

    # Display
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for res in output:
            print(f"\nImage : {res['image_path']}")
            print(f"  Plant      : {res['plant_type']}")
            print(f"  Prediction : {res['predicted_class']}")
            print(f"  Confidence : {res['confidence']*100:.2f} %")
            print(f"  Healthy    : {'✓' if res['is_healthy'] else '✗'}")
            print(f"  Top-{args.top_k} predictions:")
            for i, p in enumerate(res["top_k_predictions"], 1):
                print(f"    {i}. {p['class']:<55s} {p['confidence']*100:6.2f} %")

    # Save
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Predictions saved → {args.save}")


if __name__ == "__main__":
    main()
