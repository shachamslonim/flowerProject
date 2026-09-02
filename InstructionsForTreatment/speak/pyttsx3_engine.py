"""Offline text-to-speech via pyttsx3 (OS speech engine, SAPI5 on Windows).

Writes WAV. Needs a matching OS voice for the requested language.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXT = ".wav"


def _pick_voice(engine, language: str) -> None:
    """Select an installed voice matching ``language`` when possible.

    For Hebrew, look for a voice whose id/name mentions Hebrew; if none is
    installed, leave the default voice (it will mispronounce Hebrew) and warn.
    """
    if language != "he":
        return

    try:
        voices = engine.getProperty("voices")
    except Exception:
        return

    for voice in voices or []:
        haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
        if "hebrew" in haystack or "he-il" in haystack or "he_il" in haystack:
            engine.setProperty("voice", voice.id)
            return

    print(
        "WARNING: no Hebrew speech voice installed - the default voice will read "
        "the Hebrew text poorly. Try '--speak-engine gtts' instead.",
        file=sys.stderr,
    )


def synthesize(
    text: str, out_path: Path, language: str = "en", rate: int | None = None
) -> Path | None:
    """Write ``text`` to a WAV file at ``out_path``. Returns the path or None."""
    try:
        import pyttsx3
    except ImportError:
        print(
            "WARNING: pyttsx3 not installed - cannot read the result aloud. "
            "Install it with: pip install pyttsx3  (or use '--speak-engine gtts')",
            file=sys.stderr,
        )
        return None

    try:
        engine = pyttsx3.init()
        _pick_voice(engine, language)
        if rate is not None:
            engine.setProperty("rate", rate)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
    except Exception as e:
        print(f"WARNING: pyttsx3 text-to-speech failed ({e}).", file=sys.stderr)
        return None

    return out_path
