"""Read generated instructions aloud.

The speech engine is pluggable - pick one with the ``engine`` argument of
:func:`speak_text` (or ``--speak-engine`` on the CLI):

- ``"pyttsx3"`` - offline, uses the OS speech engine (SAPI5 on Windows). Writes
  WAV. Needs a matching OS voice for the language (e.g. a Hebrew voice for
  ``language="he"``).
- ``"gtts"``   - online (Google Translate TTS), no API key. Writes MP3. Good
  Hebrew support without installing a system voice.
- ``"mms"``    - offline neural TTS using Meta's MMS-TTS (VITS) models from
  Hugging Face (``facebook/mms-tts-eng`` / ``-heb``). Writes WAV. Needs
  ``transformers`` + ``torch``; downloads the model on first use.
"""

from .core import SPEAK_ENGINES, engine_extension, speak_text

__all__ = ["speak_text", "SPEAK_ENGINES", "engine_extension"]
