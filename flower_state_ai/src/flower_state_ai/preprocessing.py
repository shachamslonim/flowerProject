"""Stage 2: cache-resize labeled images to a manageable, common size (longest side <= 800px).

Originals are large, non-uniform DSLR/stock photos (several MB, up to ~7000px on a side).
This stage normalizes format (always re-encoded as RGB JPEG) and caps size before any
model-specific resize happens later in the training transform pipeline.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from flower_state_ai import config


@dataclass
class CacheResult:
    image_path: str
    cached_abs_path: str | None
    orig_width: int | None
    orig_height: int | None
    error: str | None


def cache_image(
    photo_root: Path,
    cache_root: Path,
    image_path: str,
    species_id: int,
    species_name: str,
    resolved_state: str,
    max_side: int = config.MAX_CACHE_SIDE,
    quality: int = 90,
) -> CacheResult:
    src = photo_root / image_path
    species_dirname = f"{species_id:03d}_{species_name}"
    dst_dir = cache_root / species_dirname / resolved_state
    # image_path (the real relative path on disk) is unique per source file, unlike its bare
    # filename - two different files can share a basename (e.g. "download.jpg"/"download.webp",
    # or the same filename physically duplicated under both closed/ and open/), which would
    # otherwise collide and silently overwrite each other in the cache.
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:8]
    dst = dst_dir / f"{Path(image_path).stem}__{digest}.jpg"

    try:
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            with Image.open(dst) as cached:
                w, h = cached.size
            return CacheResult(image_path, str(dst), w, h, None)

        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            longest = max(orig_w, orig_h)
            if longest > max_side:
                scale = max_side / longest
                new_size = (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))
                img = img.resize(new_size, Image.LANCZOS)
            dst_dir.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=quality)
        return CacheResult(image_path, str(dst), orig_w, orig_h, None)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return CacheResult(image_path, None, None, None, str(e))


def _worker(args: tuple) -> CacheResult:
    return cache_image(*args)


def build_cache(
    labels_csv: Path = config.LABELS_MASTER_CSV,
    photo_root: Path = config.PHOTO_ROOT,
    cache_root: Path = config.CACHE_DIR,
    out_csv: Path = config.LABELS_CACHED_CSV,
    errors_csv: Path = config.CACHE_ERRORS_CSV,
    max_side: int = config.MAX_CACHE_SIDE,
    max_workers: int | None = None,
) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    tasks = [
        (photo_root, cache_root, row.image_path, row.species_id, row.species_name, row.resolved_state, max_side)
        for row in labels_df.itertuples(index=False)
    ]

    max_workers = max_workers or os.cpu_count() or 4
    results: list[CacheResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())

    results_df = pd.DataFrame([asdict(r) for r in results])
    ok_df = results_df[results_df["error"].isna()].drop(columns=["error"])
    err_df = results_df[results_df["error"].notna()]

    merged = labels_df.merge(ok_df, on="image_path", how="inner")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    err_df.to_csv(errors_csv, index=False)

    print(f"[preprocessing] cached {len(merged)} images (max_side={max_side}px) -> {out_csv}")
    print(f"[preprocessing] {len(err_df)} decode/cache errors -> {errors_csv}")
    return merged


if __name__ == "__main__":
    build_cache()
