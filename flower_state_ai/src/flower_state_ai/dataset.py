"""PyTorch Dataset/DataLoader for the open/closed classifier."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flower_state_ai import config


class FlowerStateDataset(Dataset):
    """Reads from either a labels CSV path or an already-filtered DataFrame."""

    def __init__(self, source: Path | str | pd.DataFrame, transform=None):
        if isinstance(source, pd.DataFrame):
            self.df = source.reset_index(drop=True)
        else:
            self.df = pd.read_csv(source)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["cached_abs_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = config.CLASS_TO_IDX[row["resolved_state"]]
        return img, label


def get_transforms(
    train: bool,
    img_size: int = config.IMG_SIZE,
    mean: list[float] = config.IMAGENET_MEAN,
    std: list[float] = config.IMAGENET_STD,
) -> transforms.Compose:
    resize_to = round(img_size * 256 / 224)  # keep the same resize:crop ratio as the 256->224 default
    if train:
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights, to cover imbalance left after the augmentation cap."""
    counts = train_df["resolved_state"].value_counts()
    total = counts.sum()
    n_classes = len(config.CLASS_TO_IDX)
    weights = torch.zeros(n_classes)
    for state, idx in config.CLASS_TO_IDX.items():
        n = counts.get(state, 1)
        weights[idx] = total / (n_classes * n)
    return weights


def get_dataloaders(
    train_csv: Path = config.LABELS_TRAIN_FINAL_CSV,
    split_csv: Path = config.LABELS_SPLIT_CSV,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    img_size: int = config.IMG_SIZE,
    mean: list[float] = config.IMAGENET_MEAN,
    std: list[float] = config.IMAGENET_STD,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    split_df = pd.read_csv(split_csv)
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    train_dataset = FlowerStateDataset(train_csv, transform=get_transforms(True, img_size, mean, std))
    val_dataset = FlowerStateDataset(val_df, transform=get_transforms(False, img_size, mean, std))
    test_dataset = FlowerStateDataset(test_df, transform=get_transforms(False, img_size, mean, std))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
