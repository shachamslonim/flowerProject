"""Engine-agnostic entry point for reading text aloud.

Each engine lives in its own module and exposes:

    EXT: str                       # file extension it writes, e.g. ".wav"
    def synthesize(text, out_path, language, rate) -> Path | None

`synthesize` returns the path written, or ``None`` if the engine was
unavailable (it prints a warning to stderr - it must never raise).
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import gtts_engine, mms_engine, pyttsx3_engine
from .player import play_file

# name -> engine module
_ENGINES = {
    "pyttsx3": pyttsx3_engine,
    "gtts": gtts_engine,
    "mms": mms_engine,
}

SPEAK_ENGINES = tuple(_ENGINES)

# Guard against a runaway LLM result (e.g. a model that loops) being turned into
# a huge audio file. Treatment instructions are well under this.
MAX_SPEAK_CHARS = 8000


def engine_extension(engine: str) -> str:
    """File extension the given engine writes (e.g. ``.wav``)."""
    mod = _ENGINES.get((engine or "").strip().lower())
    return getattr(mod, "EXT", ".wav")


def speak_text(
    text: str,
    out_path: str | Path | None = None,
    language: str = "en",
    play: bool = True,
    engine: str = "pyttsx3",
    rate: int | None = None,
) -> Path | None:
    """Synthesize ``text`` to an audio file and optionally play it.

    Args:
        text: The text to speak.
        out_path: Where to write the audio. Defaults to ``instructions<ext>`` in
            the current directory, where ``<ext>`` depends on the engine.
        language: "en" or "he".
        play: Whether to play the audio after writing it.
        engine: Which speech engine to use - see :data:`SPEAK_ENGINES`.
        rate: Optional speech rate override (engine dependent; ignored by gtts).

    Returns:
        The path written, or ``None`` if speech was unavailable (a warning is
        printed).

    Raises:
        ValueError: If ``engine`` is not a known engine name.
    """
    key = (engine or "pyttsx3").strip().lower()
    if key not in _ENGINES:
        raise ValueError(
            f"Invalid speak engine '{engine}'. Must be one of: {', '.join(SPEAK_ENGINES)}"
        )
    mod = _ENGINES[key]

    if len(text) > MAX_SPEAK_CHARS:
        print(
            f"WARNING: text is {len(text)} chars; speaking only the first "
            f"{MAX_SPEAK_CHARS}.",
            file=sys.stderr,
        )
        text = text[:MAX_SPEAK_CHARS]

    target = Path(out_path) if out_path else Path(f"instructions{mod.EXT}")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Show exactly what is about to be spoken.
    print(
        f"--- Reading aloud ({key}, {language}) ---\n{text}\n"
        f"--- end ({len(text)} chars) ---",
        file=sys.stderr,
    )

    result = mod.synthesize(text, target, language, rate)
    if result is None:
        return None

    print(f"Audio written to: {result}", file=sys.stderr)
    if play:
        play_file(result)
    return result
