"""Stage 5: two-phase MobileNetV2 fine-tuning for open/closed classification.

Phase 1 (warmup): backbone frozen, train the classifier head only.
Phase 2 (fine-tune): unfreeze the last few MobileNetV2 blocks + head, lower LR.

Device auto-selects CUDA if available, else CPU - the same code runs unchanged on a
CPU-only machine or on a machine with an NVIDIA GPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flower_state_ai import config
from flower_state_ai.dataset import compute_class_weights, get_dataloaders
from flower_state_ai.model import build_model, freeze_backbone, unfreeze_last_n_blocks


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dl, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dl:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate_epoch(model, dl, criterion, device) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    conf = confusion_matrix(all_labels, all_preds, labels=list(config.CLASS_TO_IDX.values()))
    return total_loss / total, correct / total, macro_f1, conf


def _make_checkpoint(
    model: nn.Module,
    val_macro_f1: float,
    arch: str = "mobilenet_v2",
    img_size: int = config.IMG_SIZE,
    mean: list[float] = config.IMAGENET_MEAN,
    std: list[float] = config.IMAGENET_STD,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "class_to_idx": config.CLASS_TO_IDX,
        "arch": arch,
        "img_size": img_size,
        "normalize_mean": mean,
        "normalize_std": std,
        "val_macro_f1": val_macro_f1,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def fit(
    model: nn.Module,
    train_dl,
    val_dl,
    class_weights: torch.Tensor,
    epochs_head: int = config.PHASE1_EPOCHS,
    lr_head: float = config.PHASE1_LR,
    epochs_finetune: int = config.PHASE2_EPOCHS,
    lr_finetune: float = config.PHASE2_LR,
    unfreeze_n: int = config.UNFREEZE_LAST_N_BLOCKS,
) -> Path:
    device = get_device()
    print(f"[train] device={device}")
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log_rows: list[dict] = []
    best_f1 = -1.0
    config.BEST_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    def run_phase(phase_name: str, epochs: int, lr: float) -> None:
        nonlocal best_f1
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_loss, val_acc, val_f1, _ = evaluate_epoch(model, val_dl, criterion, device)
            scheduler.step(val_f1)
            dt = time.time() - t0

            log_rows.append({
                "phase": phase_name, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1, "seconds": dt,
            })
            print(f"[{phase_name} {epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(_make_checkpoint(model, val_f1), config.BEST_CHECKPOINT)

    freeze_backbone(model)
    run_phase("phase1_head", epochs_head, lr_head)

    unfreeze_last_n_blocks(model, unfreeze_n)
    run_phase("phase2_finetune", epochs_finetune, lr_finetune)

    torch.save(_make_checkpoint(model, best_f1), config.FINAL_CHECKPOINT)

    config.TRAIN_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(config.TRAIN_LOG_CSV, index=False)
    print(f"[train] best val macro-F1={best_f1:.4f} -> {config.BEST_CHECKPOINT}")
    return config.BEST_CHECKPOINT


def fit_from_scratch(
    model: nn.Module,
    train_dl,
    val_dl,
    class_weights: torch.Tensor,
    epochs: int = config.TINY_CNN_EPOCHS,
    lr: float = config.TINY_CNN_LR,
    weight_decay: float = config.TINY_CNN_WEIGHT_DECAY,
    arch: str = "tiny_cnn",
    img_size: int = config.TINY_CNN_IMG_SIZE,
    mean: list[float] = config.TINY_CNN_MEAN,
    std: list[float] = config.TINY_CNN_STD,
    best_checkpoint_path: Path = config.TINY_CNN_BEST_CHECKPOINT,
    final_checkpoint_path: Path = config.TINY_CNN_FINAL_CHECKPOINT,
    train_log_csv: Path = config.TINY_CNN_TRAIN_LOG_CSV,
) -> Path:
    """Single-phase training for an architecture with no pretrained backbone to protect -
    every parameter is trainable from epoch 1, reusing the same train_one_epoch/
    evaluate_epoch helpers fit() uses. AdamW (vs. fit()'s plain Adam) since weight decay
    helps a from-scratch net generalize."""
    device = get_device()
    print(f"[train] device={device} arch={arch}")
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    log_rows: list[dict] = []
    best_f1 = -1.0
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _ = evaluate_epoch(model, val_dl, criterion, device)
        scheduler.step(val_f1)
        dt = time.time() - t0

        log_rows.append({
            "phase": "from_scratch", "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1, "seconds": dt,
        })
        print(f"[from_scratch {epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(_make_checkpoint(model, val_f1, arch, img_size, mean, std), best_checkpoint_path)

    torch.save(_make_checkpoint(model, best_f1, arch, img_size, mean, std), final_checkpoint_path)

    train_log_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(train_log_csv, index=False)
    print(f"[train] best val macro-F1={best_f1:.4f} -> {best_checkpoint_path}")
    return best_checkpoint_path


def main(
    arch: str = "mobilenet_v2",
    epochs_head: int = config.PHASE1_EPOCHS,
    epochs_finetune: int = config.PHASE2_EPOCHS,
    batch_size: int | None = None,
) -> Path:
    if arch == "tiny_cnn":
        reg = config.ARCH_REGISTRY["tiny_cnn"]
        bs = batch_size or config.TINY_CNN_BATCH_SIZE
        train_dl, val_dl, _ = get_dataloaders(batch_size=bs, img_size=reg["img_size"], mean=reg["mean"], std=reg["std"])
        train_df = pd.read_csv(config.LABELS_TRAIN_FINAL_CSV)
        class_weights = compute_class_weights(train_df)
        model = build_model(arch="tiny_cnn", pretrained=False)
        return fit_from_scratch(model, train_dl, val_dl, class_weights)

    bs = batch_size or config.BATCH_SIZE
    train_dl, val_dl, _ = get_dataloaders(batch_size=bs)
    train_df = pd.read_csv(config.LABELS_TRAIN_FINAL_CSV)
    class_weights = compute_class_weights(train_df)
    model = build_model(arch="mobilenet_v2", pretrained=True)
    return fit(model, train_dl, val_dl, class_weights, epochs_head=epochs_head, epochs_finetune=epochs_finetune)


if __name__ == "__main__":
    main()
