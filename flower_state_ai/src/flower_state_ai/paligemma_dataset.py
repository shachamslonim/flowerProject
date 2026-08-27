"""Dataset/collator for PaliGemma fine-tuning: joint species-name + open/closed captioning
over species 1-23. Unlike dataset.py's per-item torchvision transform, PaliGemmaProcessor
needs to see a whole batch at once (it owns its own image resize/normalize and builds
padded, suffix-masked input_ids/labels together), so padding happens in the collator, not
per-item.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from flower_state_ai import config
from flower_state_ai.augmentation import apply_aug_op, op_tag


def species_accuracy_map(
    report_csv: Path = config.PALIGEMMA_TEST_REPORT_CSV,
    metric: str = "species_accuracy",
) -> dict[int, float]:
    """{species_id: metric value} from the most recent evaluate run's per_species_accuracy
    block. Returns an empty dict (not an error) if no report exists yet - there's nothing to
    tier before a model has ever been evaluated."""
    if not report_csv.exists():
        return {}
    with open(report_csv, encoding="utf-8") as f:
        text = f.read()
    marker = "per_species_accuracy\n"
    if marker not in text:
        return {}
    per_species_df = pd.read_csv(io.StringIO(text.split(marker, 1)[1]))
    return dict(zip(per_species_df["species_id"], per_species_df[metric]))


def _op_pool(name: str) -> list[tuple[str, float]]:
    """The three severity-tier op pools (see config.PALIGEMMA_AUG_TIERS): heavier tiers get
    more distinct augmentation *types*, not just more copies from the same narrow transform."""
    if name == "heavy":
        return (
            [("rotate", a) for a in config.ROTATION_ANGLES]
            + [("flip", 0.0)]
            + [("color_jitter", f) for f in config.PALIGEMMA_AUG_COLOR_JITTER_FACTORS]
        )
    if name == "medium":
        return (
            [("rotate", a) for a in config.ROTATION_ANGLES[:4]]
            + [("flip", 0.0)]
            + [("color_jitter", config.PALIGEMMA_AUG_COLOR_JITTER_FACTORS[0])]
        )
    return [("rotate", a) for a in config.ROTATION_ANGLES[:2]] + [("flip", 0.0)]  # "light"


def _tier_for_accuracy(accuracy: float) -> tuple[float, list[tuple[str, float]], str] | None:
    """Returns (count_multiplier, op_pool, tier_name) for the first tier whose accuracy
    ceiling the given accuracy falls under, or None if it's at/above every tier (no boost
    needed)."""
    for max_acc, multiplier, pool_name in config.PALIGEMMA_AUG_TIERS:
        if accuracy < max_acc:
            return multiplier, _op_pool(pool_name), pool_name
    return None


def build_paligemma_augmented_train(
    split_csv: Path = config.LABELS_SPLIT_CSV,
    aug_root: Path = config.PALIGEMMA_AUGMENTED_DIR,
    out_csv: Path = config.PALIGEMMA_LABELS_TRAIN_FINAL_CSV,
    report_csv: Path = config.PALIGEMMA_AUGMENTATION_REPORT_CSV,
    min_id: int = config.PALIGEMMA_SPECIES_MIN_ID,
    max_id: int = config.PALIGEMMA_SPECIES_MAX_ID,
    target_count: int = config.PALIGEMMA_AUG_TARGET_COUNT,
    max_multiplier: int = config.PALIGEMMA_AUG_MAX_MULTIPLIER,
    accuracy_map: dict[int, float] = {},
    seed: int = config.AUGMENTATION_SEED,
) -> pd.DataFrame:
    """Boosts species with few train images toward target_count using a mix of rotation,
    horizontal flip, and color-jitter variants - unlike augmentation.py's
    augment_closed_class (which balances open:closed *within* a species), this grows total
    per-species sample count, since a species-identification task needs enough examples per
    species to learn from, not just a balanced state ratio. Species already at/above
    target_count pass through untouched, UNLESS accuracy_map (from species_accuracy_map() on
    the last evaluate run) flags them as still scoring low despite having images - see
    config.PALIGEMMA_AUG_TIERS: worse accuracy gets both a bigger count multiplier AND a
    wider variety of augmentation op types, since more images alone didn't fix it but more
    augmented *variety* per image might. val/test are never touched here (caller only wires
    this into the train split - see PaliGemmaFlowerDataset)."""
    default_ops = _op_pool("heavy")
    rng = random.Random(seed)

    split_df = pd.read_csv(split_csv)
    train_df = split_df[(split_df["split"] == "train") & (split_df["species_id"].between(min_id, max_id))].copy()
    train_df["is_augmented"] = False
    train_df["source_image"] = None
    train_df["aug_op"] = None

    aug_rows: list[dict] = []
    report_rows: list[dict] = []

    for species_id, species_df in train_df.groupby("species_id"):
        species_name = species_df["species_name"].iloc[0]
        count = len(species_df)

        tier = _tier_for_accuracy(accuracy_map[species_id]) if species_id in accuracy_map else None
        if tier is not None:
            count_multiplier, ops, tier_name = tier
            species_target = max(target_count, round(count * count_multiplier))
        else:
            ops = default_ops
            tier_name = None
            species_target = target_count

        # per-image variant cap is bounded by both max_multiplier and how many distinct ops
        # this species' tier actually offers - the "light" tier's pool (3 ops) is smaller
        # than max_multiplier-1 would otherwise allow, so op_choices[i] can be shorter than
        # max_multiplier-1 and must be indexed against its own length, not the global cap.
        per_image_cap = min(len(ops), max_multiplier - 1)
        needed = max(0, species_target - count)
        max_producible = count * per_image_cap
        n_to_generate = min(needed, max_producible)

        generated = 0
        if n_to_generate > 0 and count > 0:
            rows = list(species_df.itertuples(index=False))
            variants_per_image = [0] * count
            op_choices = [rng.sample(ops, k=per_image_cap) for _ in rows]

            idx = 0
            guard = count * (max_multiplier + 5)
            while generated < n_to_generate and idx < guard:
                i = idx % count
                if variants_per_image[i] < per_image_cap:
                    row = rows[i]
                    op = op_choices[i][variants_per_image[i]]

                    src = Path(row.cached_abs_path)
                    dst_dir = aug_root / f"{species_id:03d}_{species_name}" / row.resolved_state
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    tag = op_tag(op)
                    dst = dst_dir / f"aug_{tag}_{src.name}"

                    if not dst.exists():
                        with Image.open(src) as img:
                            out_img = apply_aug_op(img.convert("RGB"), op)
                            out_img.save(dst, format="JPEG", quality=90)

                    aug_rows.append({
                        "image_path": row.image_path,
                        "species_id": species_id,
                        "species_name": species_name,
                        "resolved_state": row.resolved_state,
                        "cached_abs_path": str(dst),
                        "split": "train",
                        "is_augmented": True,
                        "source_image": str(src),
                        "aug_op": tag,
                    })
                    variants_per_image[i] += 1
                    generated += 1
                idx += 1

        report_rows.append({
            "species_id": species_id, "species_name": species_name,
            "train_count_before": count, "train_count_after": count + generated,
            "generated": generated, "target_count": species_target,
            "prior_accuracy": accuracy_map.get(species_id),
            "accuracy_tier": tier_name,
            "reached_target": (count + generated) >= species_target,
        })

    final_cols = ["image_path", "species_id", "species_name", "resolved_state",
                  "cached_abs_path", "split", "is_augmented", "source_image", "aug_op"]
    aug_df = pd.DataFrame(aug_rows)
    combined = (
        pd.concat([train_df[final_cols], aug_df[final_cols]], ignore_index=True)
        if not aug_df.empty else train_df[final_cols].copy()
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    report_df = pd.DataFrame(report_rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print(f"[paligemma-augment] train set: {len(train_df)} original + {len(aug_df)} augmented "
          f"= {len(combined)} rows -> {out_csv}")
    print(f"[paligemma-augment] per-species report -> {report_csv}")
    return combined


def build_species_vocab(
    labels_csv: Path = config.LABELS_MASTER_CSV,
    min_id: int = config.PALIGEMMA_SPECIES_MIN_ID,
    max_id: int = config.PALIGEMMA_SPECIES_MAX_ID,
) -> list[str]:
    """The fixed species-name vocabulary for species min_id..max_id, used both to validate
    the training set and to exact-match generated text at eval time."""
    df = pd.read_csv(labels_csv)
    scoped = df[df["species_id"].between(min_id, max_id)]
    names = sorted(scoped["species_name"].unique())
    return names


class PaliGemmaFlowerDataset(Dataset):
    """Reads labels_split.csv, filtered to species [min_id, max_id] and one split. Each item
    is the raw ingredients for a training/eval example - image resize/normalize and text
    tokenization both happen batch-wise in PaliGemmaCollator, not here.

    mode="joint" (default): prompt asks for species+state, target is "<species> <state>" -
    the original task, species identity unknown going in.
    mode="state_only": species name is given *in the prompt* (as the classifiers already
    have it, e.g. from folder structure) and the target is just "<state>" - tests whether
    PaliGemma's world knowledge of a named species' bloom morphology beats the classifiers
    at the one thing they're actually asked to do, rather than making it guess species too."""

    def __init__(
        self,
        split: str,
        mode: str = "joint",
        split_csv: Path | None = None,
        min_id: int = config.PALIGEMMA_SPECIES_MIN_ID,
        max_id: int = config.PALIGEMMA_SPECIES_MAX_ID,
        prompt: str = config.PALIGEMMA_PROMPT,
        state_only_prompt_template: str = config.PALIGEMMA_STATE_ONLY_PROMPT_TEMPLATE,
        limit: int | None = None,
    ):
        assert mode in ("joint", "state_only"), f"unknown mode: {mode!r}"
        if split_csv is None:
            # train reads the augmented set if build_paligemma_augmented_train() has been
            # run (boosts species with few train images); val/test always come straight
            # from labels_split.csv, unaugmented, so evaluation numbers stay honest.
            if split == "train" and config.PALIGEMMA_LABELS_TRAIN_FINAL_CSV.exists():
                split_csv = config.PALIGEMMA_LABELS_TRAIN_FINAL_CSV
            else:
                split_csv = config.LABELS_SPLIT_CSV
        df = pd.read_csv(split_csv)
        scoped = df[(df["split"] == split) & (df["species_id"].between(min_id, max_id))]
        if limit is not None:
            scoped = scoped.sample(n=min(limit, len(scoped)), random_state=config.SPLIT_SEED)
        self.df = scoped.reset_index(drop=True)
        self.mode = mode
        self.prompt = prompt
        self.state_only_prompt_template = state_only_prompt_template

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["cached_abs_path"]).convert("RGB")
        if self.mode == "state_only":
            prefix = self.state_only_prompt_template.format(species=row["species_name"])
            suffix = row["resolved_state"]
        else:
            prefix = self.prompt
            suffix = f"{row['species_name']} {row['resolved_state']}"
        return {
            "image": image,
            "prefix": prefix,
            "suffix": suffix,
            "image_path": row["image_path"],
            "species_id": int(row["species_id"]),
            "species_name": row["species_name"],
            "resolved_state": row["resolved_state"],
        }


class PaliGemmaCollator:
    """Wraps a PaliGemmaProcessor to batch-tokenize+pad. include_suffix=False for eval
    (generation only needs the prompt; passing suffix there would leak the answer into the
    input)."""

    def __init__(self, processor, include_suffix: bool = True):
        self.processor = processor
        self.include_suffix = include_suffix

    def __call__(self, batch: list[dict]) -> tuple:
        """Returns (encoding, meta) rather than folding meta into the BatchFeature - encoding
        gets encoding.to(device) called on it every step, which iterates tensor entries only;
        a raw python list of dicts under a dict key doesn't survive that cleanly."""
        images = [item["image"] for item in batch]
        # PaliGemmaProcessor will infer+prepend the image token block if omitted, but warns
        # on every call recommending it be explicit - it expects <image> literally at the
        # front of the text, not part of the image handling itself.
        prefixes = [f"<image>{item['prefix']}" for item in batch]
        kwargs = dict(images=images, text=prefixes, return_tensors="pt", padding="longest")
        if self.include_suffix:
            kwargs["suffix"] = [item["suffix"] for item in batch]
        encoding = self.processor(**kwargs)
        return encoding, batch
