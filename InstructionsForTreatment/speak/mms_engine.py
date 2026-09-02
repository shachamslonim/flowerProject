"""Offline neural text-to-speech via Meta's MMS-TTS (VITS) models on Hugging Face.

Uses ``facebook/mms-tts-eng`` / ``facebook/mms-tts-heb`` through ``transformers``.
The first run downloads the model (~145 MB each) to the HF cache; after that it
runs fully offline. Writes 16-bit PCM WAV. The ``rate`` argument is ignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXT = ".wav"

# our language code -> MMS model id
_MODELS = {
    "en": "facebook/mms-tts-eng",
    "he": "facebook/mms-tts-heb",
}

# cache loaded (model, tokenizer) per model id within a process
_CACHE: dict[str, tuple] = {}


def _load_hf_token() -> None:
    """Best-effort: put HF_TOKEN from the project .env into the environment."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and "=" in line:
            os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip().strip("\"'"))
            break


def _get_model(model_id: str):
    if model_id not in _CACHE:
        from transformers import AutoTokenizer, VitsModel

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        model.eval()
        _CACHE[model_id] = (model, tokenizer)
    return _CACHE[model_id]


def synthesize(
    text: str, out_path: Path, language: str = "en", rate: int | None = None
) -> Path | None:
    """Write ``text`` to a WAV file at ``out_path``. Returns the path or None."""
    try:
        import numpy as np
        import scipy.io.wavfile
        import torch
    except ImportError as e:
        print(
            f"WARNING: MMS-TTS needs transformers, torch, scipy and numpy ({e}). "
            "Install them or use another --speak-engine.",
            file=sys.stderr,
        )
        return None

    model_id = _MODELS.get(language)
    if model_id is None:
        print(
            f"WARNING: MMS-TTS has no configured model for language '{language}'.",
            file=sys.stderr,
        )
        return None

    _load_hf_token()

    try:
        model, tokenizer = _get_model(model_id)
    except Exception as e:
        print(
            f"WARNING: could not load MMS-TTS model '{model_id}' ({e}). "
            "The first run needs internet to download it.",
            file=sys.stderr,
        )
        return None

    if getattr(tokenizer, "is_uroman", False):
        print(
            f"WARNING: MMS model '{model_id}' needs romanized input (uroman), "
            "which is not supported here. Use '--speak-engine gtts' for this language.",
            file=sys.stderr,
        )
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform  # (1, samples), float32 in [-1, 1]
        audio = waveform.squeeze().cpu().numpy()
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(out_path), model.config.sampling_rate, pcm)
    except Exception as e:
        print(f"WARNING: MMS-TTS synthesis failed ({e}).", file=sys.stderr)
        return None

    return out_path
