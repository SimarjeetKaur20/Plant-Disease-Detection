"""
teacher_model.py
Builds the Teacher network – a pretrained ResNet-50 fine-tuned for
PlantVillage classification.

The teacher is intentionally large and accurate; it will later distil its
"soft knowledge" into the lightweight student.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from config import (
    TEACHER_MODEL_NAME, TEACHER_PRETRAINED, NUM_CLASSES,
    TEACHER_SAVE_PATH,
)
from utils import count_parameters, model_size_mb, get_logger, load_checkpoint

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature-extraction backbone registry
# ---------------------------------------------------------------------------
_BACKBONE_FACTORY = {
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
    ),

    "resnet50": (
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V1,
    ),

    "resnet101": (
        models.resnet101,
        models.ResNet101_Weights.IMAGENET1K_V1,
    ),

    "efficientnet_b3": (
        models.efficientnet_b3,
        models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    ),
}

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
class TeacherModel(nn.Module):
    """
    Wraps a pretrained CNN backbone and replaces the final classifier head
    with a two-layer MLP suited for *NUM_CLASSES* output categories.

    Architecture:
        backbone (frozen or trainable) → AdaptiveAvgPool → Dropout → FC → ReLU → FC
    """

    def __init__(
        self,
        backbone_name: str = TEACHER_MODEL_NAME,
        num_classes:   int = NUM_CLASSES,
        pretrained:    bool = TEACHER_PRETRAINED,
        freeze_backbone: bool = False,
        dropout:       float = 0.4,
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
        # Strip original classifier head; keep everything up to the pool
        # ------------------------------------------------------------------
        if backbone_name.startswith("resnet"):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone_name.startswith("efficientnet"):
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise NotImplementedError(f"Head stripping not implemented for {backbone_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone weights frozen.")

        # ------------------------------------------------------------------
        # Custom classification head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        self._init_classifier_weights()
        self._log_stats()

    # -----------------------------------------------------------------------

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_stats(self):
        total, trainable = count_parameters(self)
        mb = model_size_mb(self)
        logger.info(
            f"TeacherModel({self.backbone_name}) | "
            f"Params: {total:,} total / {trainable:,} trainable | "
            f"~{mb:.1f} MB"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for full fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen – all parameters trainable.")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def build_teacher(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = False,
) -> TeacherModel:
    """
    Build and optionally restore a TeacherModel.

    Args:
        device:           torch.device to move the model to.
        checkpoint_path:  If provided, load weights from this .pth file.
        freeze_backbone:  If True, backbone grads are disabled on creation.

    Returns:
        TeacherModel on *device*.
    """
    model = TeacherModel(freeze_backbone=freeze_backbone)

    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Teacher restored from checkpoint "
            f"(epoch {ckpt.get('epoch', '?')}, "
            f"val_acc {ckpt.get('val_acc', '?'):.4f})"
        )

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from utils import get_device
    device = get_device()
    model  = build_teacher(device)
    dummy  = torch.randn(4, 3, 224, 224).to(device)
    out    = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 15)
