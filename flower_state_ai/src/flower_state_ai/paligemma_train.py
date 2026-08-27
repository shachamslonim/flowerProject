"""Fine-tunes PaliGemma-3B via 4-bit QLoRA to jointly predict species name + open/closed
state for species 1-23 (see config.py's PALIGEMMA_* block). The memory budget this design is
built around is tight (8GB card, ~4.5GB free) - 4-bit quantization, LoRA (not full
fine-tuning), gradient checkpointing, and batch_size=1 with gradient accumulation are all
load-bearing choices, not defaults to tune away. Hand-rolled loop (not transformers.Trainer)
for explicit control over dtype casting and OOM behavior on hardware this constrained,
mirroring train.py's train_one_epoch/log_rows-to-CSV pattern.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import bitsandbytes as bnb
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from flower_state_ai import config
from flower_state_ai.paligemma_dataset import PaliGemmaCollator, PaliGemmaFlowerDataset, build_species_vocab

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_model_and_processor(lora_r: int = config.PALIGEMMA_LORA_R):
    # alpha scales with r (the standard alpha=2r convention) rather than staying fixed, so a
    # rank change actually changes LoRA capacity without also silently halving/doubling the
    # update magnitude (alpha/r) as a side effect.
    lora_alpha = 2 * lora_r
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.PALIGEMMA_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        # SDPA's fused kernel strictly requires the attention mask dtype to match the query
        # dtype; prepare_model_for_kbit_training's fp32 norm-layer upcasting (standard QLoRA
        # stability practice) leaks a float32 query into some layers under gradient
        # checkpointing, which SDPA rejects outright. Eager attention does the same math via
        # ordinary tensor ops, which tolerate mixed dtypes through normal promotion instead of
        # requiring an exact match.
        attn_implementation="eager",
    )
    processor = PaliGemmaProcessor.from_pretrained(config.PALIGEMMA_MODEL_ID)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.PALIGEMMA_LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_config)
    # LoRA A/B matrices default to float32 while the quantized base computes in bfloat16
    # (bnb_4bit_compute_dtype above) - left alone, the LoRA branch's float32 output upcasts
    # every layer it touches, and PaliGemma's attention mask (built to match the model's
    # overall bfloat16 dtype) then mismatches the now-float32 query tensor at the first
    # attention call ("Expected attn_mask dtype ... to match query dtype"). Casting the
    # trainable LoRA params to bfloat16 keeps the whole forward pass in one dtype.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    model.print_trainable_parameters()
    return model, processor


def _to_device(encoding, device: torch.device) -> dict:
    """pixel_values must match the model's bfloat16 compute dtype - the SigLIP patch-embed
    conv isn't a nn.Linear so bitsandbytes' 4-bit replacement doesn't touch it, and the image
    processor hands back float32 by default, which would otherwise mismatch at the first conv."""
    out = {}
    for k, v in encoding.items():
        if not torch.is_tensor(v):
            continue
        v = v.to(device)
        if k == "pixel_values":
            v = v.to(dtype=torch.bfloat16)
        out[k] = v
    return out


@torch.no_grad()
def _quick_val_accuracy(model, processor, val_ds, device: torch.device, mode: str, n_samples: int = 30) -> float:
    """Exact-match accuracy over a small val subsample - decoding the full val split every
    epoch at batch_size=1 is too slow on this GPU, so this is a cheap proxy to watch
    convergence; the real number comes from paligemma_evaluate.py on the test split."""
    model.eval()
    collator = PaliGemmaCollator(processor, include_suffix=False)
    n = min(n_samples, len(val_ds))
    correct = 0
    for i in range(n):
        encoding, meta = collator([val_ds[i]])
        encoding = _to_device(encoding, device)
        input_len = encoding["input_ids"].shape[1]
        generated = model.generate(**encoding, max_new_tokens=12, do_sample=False)
        text = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()
        expected = meta[0]["resolved_state"] if mode == "state_only" else f"{meta[0]['species_name']} {meta[0]['resolved_state']}"
        if text.lower() == expected.lower():
            correct += 1
    model.train()
    return correct / n


def main(
    epochs: int = config.PALIGEMMA_EPOCHS,
    lr: float = config.PALIGEMMA_LR,
    lora_r: int = config.PALIGEMMA_LORA_R,
    grad_accum_steps: int = config.PALIGEMMA_GRAD_ACCUM_STEPS,
    limit: int | None = None,
    mode: str = "joint",
) -> Path:
    assert mode in ("joint", "state_only"), f"unknown mode: {mode!r}"
    adapter_dir = config.PALIGEMMA_STATE_ONLY_ADAPTER_DIR if mode == "state_only" else config.PALIGEMMA_ADAPTER_DIR
    train_log_csv = config.PALIGEMMA_STATE_ONLY_TRAIN_LOG_CSV if mode == "state_only" else config.PALIGEMMA_TRAIN_LOG_CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paligemma-train] device={device} mode={mode}")

    vocab = build_species_vocab()
    print(f"[paligemma-train] species vocab: {len(vocab)} names -> {vocab}")

    model, processor = build_model_and_processor(lora_r=lora_r)

    train_ds = PaliGemmaFlowerDataset(split="train", mode=mode, limit=limit)
    val_ds = PaliGemmaFlowerDataset(split="val", mode=mode, limit=limit)
    print(f"[paligemma-train] train={len(train_ds)} val={len(val_ds)}")

    collator = PaliGemmaCollator(processor, include_suffix=True)
    train_loader = DataLoader(
        train_ds, batch_size=config.PALIGEMMA_BATCH_SIZE, shuffle=True, collate_fn=collator,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # PagedAdamW8bit pages optimizer state to CPU RAM under GPU pressure - meant for
    # full-model fine-tuning where that state is huge. For LoRA-only training the trainable
    # param count is tiny (~20M at r=16, well under 200MB even unquantized), so there was
    # never a real need to page anything - and on this Windows/CUDA-13.0 stack, paging
    # engaged anyway and thrashed: system RAM climbed to 9GB+ committed and per-100-step
    # throughput degraded from 0.78 to 0.21 samples/s over a few hours (confirmed via
    # Get-Process/Get-CimInstance while a run was stuck). Plain AdamW8bit keeps the (already
    # tiny) optimizer state on GPU only - no CPU round-trips, no growth over time.
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr)

    log_rows: list[dict] = []
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for step, (encoding, _meta) in enumerate(train_loader, start=1):
            encoding = _to_device(encoding, device)
            outputs = model(**encoding)
            (outputs.loss / grad_accum_steps).backward()
            running_loss += outputs.loss.item()
            n_steps += 1
            if step % 100 == 0:
                elapsed = time.time() - t0
                rate = step / elapsed
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
                print(f"[paligemma-train epoch {epoch}/{epochs}] step {step}/{len(train_loader)} "
                      f"avg_loss={running_loss / n_steps:.4f} ({rate:.2f} samples/s, {elapsed:.0f}s elapsed) "
                      f"gpu_alloc={mem_alloc:.2f}GB gpu_reserved={mem_reserved:.2f}GB alloc_retries={retries}")
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if len(train_loader) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        val_acc = _quick_val_accuracy(model, processor, val_ds, device, mode)
        dt = time.time() - t0
        avg_loss = running_loss / max(n_steps, 1)
        print(f"[paligemma-train epoch {epoch}/{epochs}] train_loss={avg_loss:.4f} "
              f"val_exact_match_sample={val_acc:.4f} ({dt:.1f}s)")
        log_rows.append({
            "epoch": epoch, "train_loss": avg_loss, "val_exact_match_sample": val_acc, "seconds": dt,
        })

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"[paligemma-train] saved adapter -> {adapter_dir}")

    train_log_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(train_log_csv, index=False)
    return adapter_dir
