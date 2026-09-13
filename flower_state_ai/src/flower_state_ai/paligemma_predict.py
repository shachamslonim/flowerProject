"""Stage 8 (PaliGemma): single-image inference, mirroring inference.py's predict() pattern
for the classifiers - this is the piece PaliGemma was missing (train.py/evaluate.py only
ever operated over a whole test-split CSV, never one arbitrary external image). Reuses
paligemma_evaluate.py's load_model/parsing rather than reimplementing them.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from flower_state_ai import config
from flower_state_ai.paligemma_dataset import PaliGemmaCollator, build_species_vocab
from flower_state_ai.paligemma_evaluate import _parse_generated, _parse_state_only, load_model


@functools.lru_cache(maxsize=2)
def _load_cached(adapter_dir: str):
    """Cached per adapter_dir (joint vs. state_only use different adapters) so the API
    process loads the 5.7GB base + adapter once, not per request."""
    return load_model(Path(adapter_dir))


def predict_paligemma(
    image_path: str,
    mode: str = "joint",
    species_name: str | None = None,
    adapter_dir: Path | None = None,
) -> dict:
    """mode="joint": returns {"species_name": str, "state": str, "raw_text": str} - species
    unknown going in, both predicted (species from the fixed species-1-24 vocabulary, or
    "UNPARSEABLE" if generation didn't match it).
    mode="state_only": species_name is required (given, not predicted, same as the
    classifiers already assume); returns {"species_name": species_name, "state": str, "raw_text": str}.
    """
    assert mode in ("joint", "state_only"), f"unknown mode: {mode!r}"
    if mode == "state_only" and not species_name:
        raise ValueError("species_name is required for mode='state_only'")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Could not read image {image_path}: {e}") from e

    if adapter_dir is None:
        adapter_dir = config.PALIGEMMA_STATE_ONLY_ADAPTER_DIR if mode == "state_only" else config.PALIGEMMA_ADAPTER_DIR
    model, processor = _load_cached(str(adapter_dir))
    device = next(model.parameters()).device

    if mode == "state_only":
        prefix = config.PALIGEMMA_STATE_ONLY_PROMPT_TEMPLATE.format(species=species_name)
    else:
        prefix = config.PALIGEMMA_PROMPT

    collator = PaliGemmaCollator(processor, include_suffix=False)
    encoding, _meta = collator([{"image": image, "prefix": prefix}])
    encoding = {k: v.to(device) for k, v in encoding.items() if torch.is_tensor(v)}
    if "pixel_values" in encoding:
        encoding["pixel_values"] = encoding["pixel_values"].to(dtype=torch.bfloat16)

    with torch.no_grad():
        input_len = encoding["input_ids"].shape[1]
        generated = model.generate(**encoding, max_new_tokens=12, do_sample=False)
        raw_text = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]

    if mode == "state_only":
        return {"species_name": species_name, "state": _parse_state_only(raw_text), "raw_text": raw_text}

    vocab = build_species_vocab()
    species_vocab_lower = {name.lower(): name for name in vocab}
    species_pred, state_pred = _parse_generated(raw_text, species_vocab_lower)
    return {"species_name": species_pred, "state": state_pred, "raw_text": raw_text}
