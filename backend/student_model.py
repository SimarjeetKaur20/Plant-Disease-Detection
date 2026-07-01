"""
student_model.py
Lightweight Student network based on MobileNetV2.

The student is designed to be fast and small enough for edge / mobile
deployment while retaining high accuracy via knowledge distillation from
the Teacher (ResNet-50).
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from config import (
    STUDENT_MODEL_NAME, STUDENT_PRETRAINED, NUM_CLASSES,
    STUDENT_SAVE_PATH,
)
from utils import count_parameters, model_size_mb, get_logger, load_checkpoint

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backbone registry  (lightweight models only)
# ---------------------------------------------------------------------------
_BACKBONE_FACTORY = {
    "mobilenet_v2": (
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "shufflenet_v2_x1_0": (
        models.shufflenet_v2_x1_0,
        models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}


# ---------------------------------------------------------------------------
# Intermediate feature hook (used for feature-map distillation)
# ---------------------------------------------------------------------------
class FeatureExtractorHook:
    """Captures intermediate feature maps via a forward hook."""

    def __init__(self):
        self.features: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, layer: nn.Module):
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output):
        self.features = output

    def remove(self):
        if self._handle:
            self._handle.remove()


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------
class StudentModel(nn.Module):
    """
    Thin wrapper around a pretrained lightweight backbone with a custom
    classification head matching the teacher's output dimensionality.

    The head mirrors the teacher's head so that logit-level distillation
    (soft targets) works naturally.
    """

    def __init__(
        self,
        backbone_name: str  = STUDENT_MODEL_NAME,
        num_classes:   int  = NUM_CLASSES,
        pretrained:    bool = STUDENT_PRETRAINED,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name not in _BACKBONE_FACTORY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from {list(_BACKBONE_FACTORY.keys())}"
            )

        builder, weights = _BACKBONE_FACTORY[backbone_name]
        base_model = builder(weights=weights if pretrained else None)

        # ------------------------------------------------------------------
        # Strip original classifier
        # ------------------------------------------------------------------
        if backbone_name.startswith("mobilenet_v2"):
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        elif backbone_name.startswith("mobilenet_v3"):
            in_features = base_model.classifier[0].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        elif backbone_name.startswith("shufflenet"):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone_name.startswith("efficientnet"):
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise NotImplementedError(f"Head stripping not implemented for {backbone_name}")

        # ------------------------------------------------------------------
        # Custom head  (matches teacher's head structure)
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_head()
        self._log_stats()

    # -----------------------------------------------------------------------

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_stats(self):
        total, trainable = count_parameters(self)
        mb = model_size_mb(self)
        logger.info(
            f"StudentModel({self.backbone_name}) | "
            f"Params: {total:,} total / {trainable:,} trainable | "
            f"~{mb:.1f} MB"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # MobileNetV2 returns (B, C, 1, 1) – flatten if needed
        if features.dim() > 2:
            features = features.flatten(1)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_student(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> StudentModel:
    """
    Build and optionally restore a StudentModel.

    Args:
        device:           torch device.
        checkpoint_path:  Optional .pth file to restore weights from.

    Returns:
        StudentModel moved to *device*.
    """
    model = StudentModel()

    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Student restored from checkpoint "
            f"(epoch {ckpt.get('epoch', '?')}, "
            f"val_acc {ckpt.get('val_acc', '?'):.4f})"
        )

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import get_device
    device = get_device()
    model  = build_student(device)
    dummy  = torch.randn(4, 3, 224, 224).to(device)
    out    = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 15)
