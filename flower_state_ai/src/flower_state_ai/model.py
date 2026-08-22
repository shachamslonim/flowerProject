"""Model builders and freeze/unfreeze helpers for open/closed classification.

build_model(arch=...) dispatches to either the ImageNet-pretrained MobileNetV2 (fine-tuned
via freeze_backbone/unfreeze_last_n_blocks below) or the tiny_model.TinyFlowerCNN trained
from scratch (see tiny_model.py) - the two architectures this project trains.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V2_Weights

from flower_state_ai import config
from flower_state_ai.tiny_model import build_tiny_model


def build_model(arch: str = "mobilenet_v2", pretrained: bool = True) -> nn.Module:
    if arch == "tiny_cnn":
        if pretrained:
            print("[model] tiny_cnn has no pretrained weights - training from scratch")
        return build_tiny_model()
    if arch != "mobilenet_v2":
        raise ValueError(f"Unknown arch: {arch!r}")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(config.CLASS_TO_IDX))
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: train only the classification head."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int = config.UNFREEZE_LAST_N_BLOCKS) -> None:
    """Phase 2: fine-tune the last n inverted-residual blocks plus the classifier head."""
    for param in model.features.parameters():
        param.requires_grad = False
    for block in model.features[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
