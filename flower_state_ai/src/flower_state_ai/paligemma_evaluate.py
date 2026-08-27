"""Evaluates a fine-tuned PaliGemma adapter on the species-1-23 test split, greedily
generating "<species> <state>" text per image and deterministically parsing it back into
(species_pred, state_pred) for per-species accuracy reporting - mirrors evaluate.py's
overall-metrics-then-per_species_accuracy report shape.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from flower_state_ai import config
from flower_state_ai.paligemma_dataset import PaliGemmaCollator, PaliGemmaFlowerDataset, build_species_vocab


def load_model(adapter_dir: Path = config.PALIGEMMA_ADAPTER_DIR):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        config.PALIGEMMA_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",  # see paligemma_train.py's build_model_and_processor
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    processor = PaliGemmaProcessor.from_pretrained(adapter_dir)
    return model, processor


def _parse_generated(text: str, species_vocab_lower: dict[str, str]) -> tuple[str, str]:
    """Deterministic, no fuzzy matching - the whole point of the fixed '<species> <state>'
    target format is that this parse either cleanly succeeds or clearly fails as
    UNPARSEABLE, never a silent near-miss."""
    text = text.strip()
    if " " not in text:
        return "UNPARSEABLE", "UNPARSEABLE"
    species_part, state_part = text.rsplit(" ", 1)
    state_pred = state_part.lower() if state_part.lower() in ("open", "closed") else "UNPARSEABLE"
    species_pred = species_vocab_lower.get(species_part.strip().lower(), "UNPARSEABLE")
    return species_pred, state_pred


def _parse_state_only(text: str) -> str:
    """state_only mode's target is just 'open'/'closed' - no species to split off."""
    text = text.strip().lower()
    return text if text in ("open", "closed") else "UNPARSEABLE"


def evaluate(adapter_dir: Path | None = None, mode: str = "joint") -> pd.DataFrame:
    assert mode in ("joint", "state_only"), f"unknown mode: {mode!r}"
    if adapter_dir is None:
        adapter_dir = config.PALIGEMMA_STATE_ONLY_ADAPTER_DIR if mode == "state_only" else config.PALIGEMMA_ADAPTER_DIR
    test_report_csv = config.PALIGEMMA_STATE_ONLY_TEST_REPORT_CSV if mode == "state_only" else config.PALIGEMMA_TEST_REPORT_CSV
    state_confusion_csv = config.PALIGEMMA_STATE_ONLY_STATE_CONFUSION_CSV if mode == "state_only" else config.PALIGEMMA_STATE_CONFUSION_CSV
    misclassified_csv = config.PALIGEMMA_STATE_ONLY_MISCLASSIFIED_CSV if mode == "state_only" else config.PALIGEMMA_MISCLASSIFIED_CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(adapter_dir)

    vocab = build_species_vocab()
    species_vocab_lower = {name.lower(): name for name in vocab}

    test_ds = PaliGemmaFlowerDataset(split="test", mode=mode)
    collator = PaliGemmaCollator(processor, include_suffix=False)
    print(f"[paligemma-evaluate] device={device} mode={mode} test={len(test_ds)}")

    rows = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            encoding, meta = collator([test_ds[i]])
            encoding = {k: v.to(device) for k, v in encoding.items() if torch.is_tensor(v)}
            if "pixel_values" in encoding:
                encoding["pixel_values"] = encoding["pixel_values"].to(dtype=torch.bfloat16)
            input_len = encoding["input_ids"].shape[1]
            generated = model.generate(**encoding, max_new_tokens=12, do_sample=False)
            raw_text = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]

            m = meta[0]
            if mode == "state_only":
                state_pred = _parse_state_only(raw_text)
                species_pred = m["species_name"]  # given, not predicted - always "correct"
            else:
                species_pred, state_pred = _parse_generated(raw_text, species_vocab_lower)

            rows.append({
                "image_path": m["image_path"],
                "species_id": m["species_id"],
                "species_name": m["species_name"],
                "resolved_state": m["resolved_state"],
                "raw_generated_text": raw_text,
                "species_pred": species_pred,
                "state_pred": state_pred,
                "species_correct": species_pred == m["species_name"],
                "state_correct": state_pred == m["resolved_state"],
            })
            if (i + 1) % 100 == 0:
                print(f"[paligemma-evaluate] {i + 1}/{len(test_ds)}")

    df = pd.DataFrame(rows)
    df["joint_correct"] = df["species_correct"] & df["state_correct"]
    df["unparseable"] = (df["species_pred"] == "UNPARSEABLE") | (df["state_pred"] == "UNPARSEABLE")

    overall = pd.DataFrame([{
        "species_exact_match_rate": df["species_correct"].mean(),
        "state_exact_match_rate": df["state_correct"].mean(),
        "joint_exact_match_rate": df["joint_correct"].mean(),
        "unparseable_rate": df["unparseable"].mean(),
        "n_test": len(df),
    }])

    per_species = (
        df.groupby(["species_id", "species_name"])
        .agg(
            n_test=("image_path", "size"),
            species_accuracy=("species_correct", "mean"),
            state_accuracy=("state_correct", "mean"),
            joint_accuracy=("joint_correct", "mean"),
        )
        .reset_index()
        .sort_values("species_id")
    )

    parseable = df[~df["unparseable"]]
    state_confusion = pd.crosstab(
        parseable["resolved_state"], parseable["state_pred"], dropna=False,
    ).reindex(index=["closed", "open"], columns=["closed", "open"], fill_value=0)

    misclassified = (
        df[~df["joint_correct"]]
        .sort_values("species_id")[[
            "image_path", "species_name", "resolved_state", "raw_generated_text",
            "species_pred", "state_pred",
        ]]
    )

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(test_report_csv, index=False)
    with open(test_report_csv, "a", encoding="utf-8") as f:
        f.write("\nper_species_accuracy\n")
    per_species.to_csv(test_report_csv, mode="a", index=False)
    state_confusion.to_csv(state_confusion_csv)
    misclassified.to_csv(misclassified_csv, index=False)

    print(f"[paligemma-evaluate] mode={mode} species_exact_match={overall['species_exact_match_rate'][0]:.4f} "
          f"state_exact_match={overall['state_exact_match_rate'][0]:.4f} "
          f"joint_exact_match={overall['joint_exact_match_rate'][0]:.4f} "
          f"unparseable_rate={overall['unparseable_rate'][0]:.4f}")
    print(f"[paligemma-evaluate] wrote {test_report_csv}, {state_confusion_csv}, {misclassified_csv}")
    return per_species
