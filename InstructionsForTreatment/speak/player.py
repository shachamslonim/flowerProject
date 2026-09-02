"""Play an audio file, tolerating a missing audio device or player."""

from __future__ import annotations

import sys
from pathlib import Path


def play_file(path: Path) -> None:
    """Play ``path``. WAV uses winsound on Windows; anything else opens in the
    default player. Never raises - prints a warning on failure.
    """
    if path.suffix.lower() == ".wav":
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return
        except Exception:
            pass

    try:
        import os

        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"WARNING: could not play audio ({e}).", file=sys.stderr)
