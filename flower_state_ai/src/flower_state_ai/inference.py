"""Stage 8: single-image inference wrapper - this is the function the LangChain agent tool calls."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from flower_state_ai import config
from flower_state_ai.dataset import get_transforms
from flower_state_ai.model import build_model


@dataclass
class PredictionResult:
    label: str
    confidence: float


@functools.lru_cache(maxsize=4)
def load_model(checkpoint_path: str = str(config.BEST_CHECKPOINT), device: str = "cpu", arch: str = "mobilenet_v2") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(arch=ckpt.get("arch", arch), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def predict(image_path: str, model: torch.nn.Module | None = None, device: str = "cpu", arch: str = "mobilenet_v2") -> PredictionResult:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reg = config.ARCH_REGISTRY[arch]
    model = model or load_model(str(reg["best_checkpoint"]), device=device, arch=arch)
    transform = get_transforms(False, reg["img_size"], reg["mean"], reg["std"])
    idx_to_class = {v: k for k, v in config.CLASS_TO_IDX.items()}

    try:
        img = Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Could not read image {image_path}: {e}") from e

    with torch.no_grad():
        x = transform(img).unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=1)[0]
        pred_idx = int(probs.argmax())

    return PredictionResult(label=idx_to_class[pred_idx], confidence=float(probs[pred_idx]))


def predict_tiny(
    image_path: str,
    onnx_path: str | None = None,
    img_size: int | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    arch: str = "mobilenet_v2",
) -> PredictionResult:
    """Same as predict(), but runs the exported ONNX int8 'ultra tiny' export instead.
    Defaults to mobilenet_v2's export; pass arch="tiny_cnn" for that architecture's own
    (already tiny) ONNX int8 export - onnx_path/img_size/mean/std default from
    ARCH_REGISTRY[arch] unless overridden explicitly."""
    import numpy as np
    import onnxruntime as ort

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    reg = config.ARCH_REGISTRY[arch]
    onnx_path = onnx_path or str(reg["onnx_int8_path"])
    transform = get_transforms(False, img_size or reg["img_size"], mean or reg["mean"], std or reg["std"])
    idx_to_class = {v: k for k, v in config.CLASS_TO_IDX.items()}

    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).numpy().astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    pred_idx = int(probs.argmax())

    return PredictionResult(label=idx_to_class[pred_idx], confidence=float(probs[pred_idx]))
