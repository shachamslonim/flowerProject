"""Online text-to-speech via gTTS (Google Translate TTS). No API key.

Writes MP3. Needs internet access at run time. Handles Hebrew well without any
OS voice installed. The ``rate`` argument is not supported and is ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXT = ".mp3"

# gTTS language codes differ from the module's "en"/"he".
_LANG_MAP = {"en": "en", "he": "iw"}


def synthesize(
    text: str, out_path: Path, language: str = "en", rate: int | None = None
) -> Path | None:
    """Write ``text`` to an MP3 file at ``out_path``. Returns the path or None."""
    try:
        from gtts import gTTS
    except ImportError:
        print(
            "WARNING: gTTS not installed - cannot read the result aloud. "
            "Install it with: pip install gTTS  (or use '--speak-engine pyttsx3')",
            file=sys.stderr,
        )
        return None

    lang = _LANG_MAP.get(language, language)
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(str(out_path))
    except Exception as e:
        print(
            f"WARNING: gTTS text-to-speech failed ({e}). "
            "It needs an internet connection.",
            file=sys.stderr,
        )
        return None

    return out_path
