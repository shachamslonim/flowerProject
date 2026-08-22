"""Stage 3: stratified train/val/test split at the source-image level.

Split happens BEFORE augmentation on purpose - rotated "closed" images are near-duplicates
of their source, so augmenting first risks a source photo and its rotated clone landing in
different splits, leaking information into "unseen" val/test data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from flower_state_ai import config


def stratified_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_RATIO,
    val_size: float = config.VAL_RATIO,
    seed: int = config.SPLIT_SEED,
    min_per_cell: int = config.MIN_PER_CELL,
) -> pd.DataFrame:
    df = df.copy()
    df["strata_key"] = df["species_id"].astype(str) + "_" + df["resolved_state"]

    counts = df["strata_key"].value_counts()
    small_cells = counts[counts < min_per_cell].index
    small_df = df[df["strata_key"].isin(small_cells)].copy()
    big_df = df[~df["strata_key"].isin(small_cells)].copy()

    if not small_df.empty:
        small_df["split"] = "train"
        print(f"[splitting] {len(small_df)} rows in {len(small_cells)} small strata "
              f"(<{min_per_cell} images) forced into train split")

    if big_df.empty:
        return pd.concat([small_df, big_df], ignore_index=True).drop(columns=["strata_key"])

    train_df, temp_df = train_test_split(
        big_df, test_size=(val_size + test_size), stratify=big_df["strata_key"], random_state=seed,
    )
    train_df = train_df.copy()
    train_df["split"] = "train"

    # A stratum can pass the min_per_cell filter yet still leave < 2 members in temp_df after
    # the first split (e.g. a stratum of exactly 3 -> 2 train / 1 temp), which a second
    # stratified split can't handle. Those leftover singletons go straight to val.
    temp_counts = temp_df["strata_key"].value_counts()
    temp_small_keys = temp_counts[temp_counts < 2].index
    temp_small_df = temp_df[temp_df["strata_key"].isin(temp_small_keys)].copy()
    temp_big_df = temp_df[~temp_df["strata_key"].isin(temp_small_keys)].copy()

    if not temp_small_df.empty:
        temp_small_df["split"] = "val"
        print(f"[splitting] {len(temp_small_df)} rows in {len(temp_small_keys)} strata "
              f"with a single val/test candidate forced into val split")

    if temp_big_df.empty:
        val_df = temp_small_df
        test_df = pd.DataFrame(columns=temp_df.columns)
    else:
        relative_test = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_big_df, test_size=relative_test, stratify=temp_big_df["strata_key"], random_state=seed,
        )
        val_df = pd.concat([val_df.copy(), temp_small_df], ignore_index=True)
        test_df = test_df.copy()

    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df, small_df], ignore_index=True)
    return result.drop(columns=["strata_key"])


def build_split(
    labels_cached_csv: Path = config.LABELS_CACHED_CSV,
    out_csv: Path = config.LABELS_SPLIT_CSV,
    summary_csv: Path = config.SPLIT_SUMMARY_CSV,
) -> pd.DataFrame:
    df = pd.read_csv(labels_cached_csv)
    split_df = stratified_split(df)

    summary = (
        split_df.groupby(["species_id", "species_name", "resolved_state", "split"])
        .size()
        .reset_index(name="count")
        .sort_values(["species_id", "resolved_state", "split"])
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    overall = split_df["split"].value_counts(normalize=True).round(3).to_dict()
    print(f"[splitting] wrote {len(split_df)} rows -> {out_csv}")
    print(f"[splitting] overall split ratios: {overall}")
    return split_df


if __name__ == "__main__":
    build_split()
