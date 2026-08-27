"""Stage 4: rotation + scale augmentation to reduce open/closed imbalance in the train split.

Only the train split's "closed" images are augmented (val/test stay untouched and
un-augmented so evaluation numbers are honest). Each source image can produce up to
AUGMENTATION_MAX_MULTIPLIER - 1 variants, drawn from a combined pool of rotation angles
(ROTATION_ANGLES) and zoom levels (SCALE_FACTORS: <1.0 crops in, >1.0 zooms out via
reflect-padding) - mixing both keeps generated variants more distinct from each other than
rotation alone once a species needs many variants per source image. Generation stops once a
species' closed:open ratio in train approaches AUGMENTATION_TARGET_RATIO or the cap is hit,
whichever comes first. Any residual imbalance after the cap is handled by class-weighted
loss during training (see dataset.compute_class_weights).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

from flower_state_ai import config

AugOp = tuple[str, float]


def _largest_rotated_rect(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle (in the original, unrotated frame) that fits inside
    a w x h image after it has been rotated by angle_rad, with no border padding visible."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        if width_is_longer:
            wr, hr = x / sin_a, x / cos_a
        else:
            wr, hr = x / cos_a, x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate_and_crop(img: Image.Image, angle: float) -> Image.Image:
    """Rotate then center-crop to the largest border-free rectangle, resized back to the
    original dimensions - avoids the black-corner artifacts a naive rotate() would leave."""
    w, h = img.size
    rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated.size
    crop_w, crop_h = _largest_rotated_rect(w, h, math.radians(angle))
    crop_w = max(1.0, min(crop_w, rw))
    crop_h = max(1.0, min(crop_h, rh))
    left = (rw - crop_w) / 2
    top = (rh - crop_h) / 2
    cropped = rotated.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.LANCZOS)


def scale_and_crop(img: Image.Image, scale: float) -> Image.Image:
    """scale < 1.0: zoom in by center-cropping to scale*w x scale*h then resizing back up
    (simulates a closer-framed photo). scale > 1.0: zoom out by shrinking the whole image
    to 1/scale size then reflect-padding back to the original dimensions (simulates a
    photo taken further back, without the black borders a naive canvas-paste would leave)."""
    w, h = img.size
    if scale <= 1.0:
        crop_w, crop_h = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (w - crop_w) / 2, (h - crop_h) / 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.LANCZOS)

    inv = 1.0 / scale
    small_w, small_h = max(1, round(w * inv)), max(1, round(h * inv))
    shrunk = img.resize((small_w, small_h), Image.LANCZOS)
    arr = np.array(shrunk)
    pad_h, pad_w = h - small_h, w - small_w
    pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    padded = np.pad(arr, pad, mode="reflect")
    return Image.fromarray(padded[:h, :w])


def flip_horizontal(img: Image.Image, _param: float) -> Image.Image:
    return ImageOps.mirror(img)


def color_jitter(img: Image.Image, factor: float) -> Image.Image:
    """factor scales brightness/contrast/color together, each perturbed by an independent
    random draw around it, so two calls with the same factor still produce visibly different
    variants - useful when a rare species needs several color-jitter copies from one source."""
    rng = random.Random(hash((id(img), factor)) & 0xFFFFFFFF)
    out = img
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        jitter = rng.uniform(min(factor, 1 / factor), max(factor, 1 / factor))
        out = enhancer_cls(out).enhance(jitter)
    return out


_OP_APPLIERS = {
    "rotate": rotate_and_crop,
    "scale": scale_and_crop,
    "flip": flip_horizontal,
    "color_jitter": color_jitter,
}


def apply_aug_op(img: Image.Image, op: AugOp) -> Image.Image:
    op_type, param = op
    return _OP_APPLIERS[op_type](img, param)


def op_tag(op: AugOp) -> str:
    op_type, param = op
    return f"{op_type}{param:g}"


def augment_closed_class(
    train_df: pd.DataFrame,
    aug_root: Path,
    target_ratio: float = config.AUGMENTATION_TARGET_RATIO,
    max_multiplier: int = config.AUGMENTATION_MAX_MULTIPLIER,
    angles: list[int] = config.ROTATION_ANGLES,
    scale_factors: list[float] = config.SCALE_FACTORS,
    seed: int = config.AUGMENTATION_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ops: list[AugOp] = [("rotate", a) for a in angles] + [("scale", s) for s in scale_factors]
    rng = random.Random(seed)
    aug_rows: list[dict] = []
    report_rows: list[dict] = []

    for species_id, species_df in train_df.groupby("species_id"):
        species_name = species_df["species_name"].iloc[0]
        closed_df = species_df[species_df["resolved_state"] == "closed"]
        open_count = int((species_df["resolved_state"] == "open").sum())
        closed_count = len(closed_df)

        target_closed = int(round(open_count * target_ratio))
        needed = max(0, target_closed - closed_count)
        max_producible = closed_count * (max_multiplier - 1)
        n_to_generate = min(needed, max_producible)

        generated = 0
        if n_to_generate > 0 and closed_count > 0:
            closed_rows = list(closed_df.itertuples(index=False))
            variants_per_image = [0] * closed_count
            op_choices = [rng.sample(ops, k=min(len(ops), max_multiplier - 1)) for _ in closed_rows]

            idx = 0
            guard = closed_count * (max_multiplier + 5)
            while generated < n_to_generate and idx < guard:
                i = idx % closed_count
                if variants_per_image[i] < (max_multiplier - 1):
                    row = closed_rows[i]
                    op = op_choices[i][variants_per_image[i]]

                    src = Path(row.cached_abs_path)
                    dst_dir = aug_root / f"{species_id:03d}_{species_name}" / "closed"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / f"aug_{op_tag(op)}_{src.name}"

                    if not dst.exists():
                        with Image.open(src) as img:
                            out_img = apply_aug_op(img.convert("RGB"), op)
                            out_img.save(dst, format="JPEG", quality=90)

                    aug_rows.append({
                        "image_path": row.image_path,
                        "species_id": species_id,
                        "species_name": species_name,
                        "resolved_state": "closed",
                        "cached_abs_path": str(dst),
                        "split": "train",
                        "is_augmented": True,
                        "source_image": str(src),
                        "aug_op": op[0],
                        "aug_param": op[1],
                    })
                    variants_per_image[i] += 1
                    generated += 1
                idx += 1

        report_rows.append({
            "species_id": species_id,
            "species_name": species_name,
            "open_train_count": open_count,
            "closed_train_count_before": closed_count,
            "closed_train_count_after": closed_count + generated,
            "generated": generated,
            "target_closed": target_closed,
            "reached_target": (closed_count + generated) >= target_closed,
        })

    aug_df = pd.DataFrame(aug_rows)
    report_df = pd.DataFrame(report_rows)
    return aug_df, report_df


def build_augmented_train_set(
    labels_split_csv: Path = config.LABELS_SPLIT_CSV,
    aug_root: Path = config.AUGMENTED_DIR,
    out_csv: Path = config.LABELS_TRAIN_FINAL_CSV,
    report_csv: Path = config.AUGMENTATION_REPORT_CSV,
) -> pd.DataFrame:
    split_df = pd.read_csv(labels_split_csv)
    train_df = split_df[split_df["split"] == "train"].copy()
    train_df["is_augmented"] = False
    train_df["source_image"] = None
    train_df["aug_op"] = None
    train_df["aug_param"] = None

    aug_df, report_df = augment_closed_class(train_df, aug_root)

    final_cols = ["image_path", "species_id", "species_name", "resolved_state",
                  "cached_abs_path", "split", "is_augmented", "source_image", "aug_op", "aug_param"]
    if aug_df.empty:
        combined = train_df[final_cols].copy()
    else:
        combined = pd.concat([train_df[final_cols], aug_df[final_cols]], ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print(f"[augmentation] train set: {len(train_df)} original + {len(aug_df)} augmented "
          f"= {len(combined)} rows -> {out_csv}")
    print(f"[augmentation] per-species report -> {report_csv}")
    return combined


if __name__ == "__main__":
    build_augmented_train_set()
