"""A genuinely tiny CNN, trained from scratch, for microcontroller-class ("ultra tiny")
deployment - distinct from MobileNetV2 export_tiny.py quantization, which shrinks an
ImageNet-pretrained backbone but still lands in the megabytes. This is a small
depthwise-separable-conv net in the style of TFLite-Micro person-detection models: 96x96
input, tens of thousands of params, no pretrained weights.
"""
from __future__ import annotations

import torch.nn as nn

from flower_state_ai import config


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class TinyFlowerCNN(nn.Module):
    """Stem conv + 7 depthwise-separable blocks (4 downsampling + 3 same-resolution
    refinement blocks added at the deeper stages, since the input is already down to 6x6
    by the last downsample and further striding would leave too little spatial signal for
    global average pooling) + global average pool + linear head. Channel progression
    8->16->32->64->96 - a wider 16->32->64->128 variant was measured to add 2.6x the
    params/1.7x the export size for no meaningful accuracy gain (and more overfitting) on
    this dataset size, so this stayed the narrower config. Input expected at
    config.TINY_CNN_IMG_SIZE (96px)."""

    def __init__(self, num_classes: int = 2, width_mult: float = 1.0):
        super().__init__()

        def c(channels: int) -> int:
            return _make_divisible(channels * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(8), 3, 2, 1, bias=False),
            nn.BatchNorm2d(c(8)),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(c(8), c(16), stride=2),
            DepthwiseSeparableConv(c(16), c(32), stride=2),
            DepthwiseSeparableConv(c(32), c(32), stride=1),
            DepthwiseSeparableConv(c(32), c(64), stride=2),
            DepthwiseSeparableConv(c(64), c(64), stride=1),
            DepthwiseSeparableConv(c(64), c(96), stride=1),
            DepthwiseSeparableConv(c(96), c(96), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c(96), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_tiny_model(
    num_classes: int = len(config.CLASS_TO_IDX),
    width_mult: float = config.TINY_CNN_WIDTH_MULT,
) -> nn.Module:
    model = TinyFlowerCNN(num_classes=num_classes, width_mult=width_mult)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[tiny_model] TinyFlowerCNN params: {n_params:,}")
    return model
