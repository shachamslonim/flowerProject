"""Stage 7: export a quantized / ONNX 'ultra tiny' deployable artifact.

MobileNetV2 is almost entirely Conv2d, so plain torch.quantization.quantize_dynamic
(which only touches nn.Linear/nn.LSTM by default) barely shrinks the conv backbone - it's
included because it was explicitly asked for, but the real size win comes from the ONNX
export + onnxruntime int8 dynamic quantization path, which does quantize the conv/matmul
weights. Both are produced so the actual size/accuracy tradeoff can be compared.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from flower_state_ai import config
from flower_state_ai.dataset import get_transforms
from flower_state_ai.evaluate import load_checkpoint


def quantize_torch_dynamic(model: nn.Module, out_path: Path = config.TORCH_QUANT_PATH) -> tuple[Path, nn.Module]:
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, out_path)  # whole-module save: dynamic-quantized layers aren't plain state_dict-reloadable
    return out_path, quantized


def export_onnx(model: nn.Module, out_path: Path = config.ONNX_FP32_PATH, img_size: int = config.IMG_SIZE) -> Path:
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}}, opset_version=17,
        dynamo=False,  # the newer dynamo=True exporter (torch>=2.9 default) produces a graph
                       # onnxruntime.quantization's shape inference chokes on for this model
    )
    return out_path


def quantize_onnx_dynamic(onnx_path: Path = config.ONNX_FP32_PATH, out_path: Path = config.ONNX_INT8_PATH) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def _file_size_mb(path: Path) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 3)


def _sample_val_df(sample_size: int) -> pd.DataFrame:
    split_df = pd.read_csv(config.LABELS_SPLIT_CSV)
    val_df = split_df[split_df["split"] == "val"]
    if len(val_df) > sample_size:
        val_df = val_df.sample(n=sample_size, random_state=config.SPLIT_SEED)
    return val_df


def _accuracy_torch(model: nn.Module, val_df: pd.DataFrame, transform) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for row in val_df.itertuples(index=False):
            img = Image.open(row.cached_abs_path).convert("RGB")
            x = transform(img).unsqueeze(0)
            pred = int(model(x).argmax(1).item())
            correct += int(pred == config.CLASS_TO_IDX[row.resolved_state])
            total += 1
    return correct / total if total else 0.0


def _accuracy_onnx(onnx_path: Path, val_df: pd.DataFrame, transform) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    correct, total = 0, 0
    for row in val_df.itertuples(index=False):
        img = Image.open(row.cached_abs_path).convert("RGB")
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
        logits = session.run(None, {input_name: x})[0]
        pred = int(logits.argmax())
        correct += int(pred == config.CLASS_TO_IDX[row.resolved_state])
        total += 1
    return correct / total if total else 0.0


def run_export(
    checkpoint_path: Path | None = None,
    accuracy_sample_size: int = 200,
    arch: str = "mobilenet_v2",
) -> pd.DataFrame:
    reg = config.ARCH_REGISTRY[arch]
    checkpoint_path = checkpoint_path or reg["best_checkpoint"]
    model = load_checkpoint(checkpoint_path, arch=arch)
    transform = get_transforms(False, reg["img_size"], reg["mean"], reg["std"])
    val_df = _sample_val_df(accuracy_sample_size)

    torch_quant_path, quant_model = quantize_torch_dynamic(model, out_path=reg["torch_quant_path"])
    onnx_fp32_path = export_onnx(model, out_path=reg["onnx_fp32_path"], img_size=reg["img_size"])
    onnx_int8_path = quantize_onnx_dynamic(onnx_fp32_path, out_path=reg["onnx_int8_path"])

    rows = [
        {
            "variant": "torch_fp32_checkpoint",
            "path": str(checkpoint_path),
            "size_mb": _file_size_mb(checkpoint_path),
            "val_accuracy": _accuracy_torch(model, val_df, transform),
        },
        {
            "variant": "torch_dynamic_quant",
            "path": str(torch_quant_path),
            "size_mb": _file_size_mb(torch_quant_path),
            "val_accuracy": _accuracy_torch(quant_model, val_df, transform),
        },
        {
            "variant": "onnx_fp32",
            "path": str(onnx_fp32_path),
            "size_mb": _file_size_mb(onnx_fp32_path),
            "val_accuracy": _accuracy_onnx(onnx_fp32_path, val_df, transform),
        },
        {
            "variant": "onnx_int8",
            "path": str(onnx_int8_path),
            "size_mb": _file_size_mb(onnx_int8_path),
            "val_accuracy": _accuracy_onnx(onnx_int8_path, val_df, transform),
        },
    ]
    comparison_df = pd.DataFrame(rows)
    comparison_csv = reg["model_size_comparison_csv"]
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_csv, index=False)
    print(comparison_df.to_string(index=False))
    print(f"[export_tiny] arch={arch} wrote {comparison_csv}")
    return comparison_df


if __name__ == "__main__":
    run_export()
