"""Machine-translate text with deep-translator (Google Translate - free, no key).

Used to turn grounded English LLM output into Hebrew, because the small local
Ollama models cannot write Hebrew reliably. Best-effort: on any failure it warns
to stderr and returns the original text unchanged.
"""

from __future__ import annotations

import sys
import time

# deep-translator's Google backend uses legacy codes - Hebrew is "iw".
_LANG_CODES = {"he": "iw", "iw": "iw", "en": "en"}

# GoogleTranslator rejects very long strings; translate paragraph by paragraph
# when the whole text is bigger than this.
_MAX_SINGLE_CALL = 4000


def _translate_once(translator, text: str, retries: int) -> str | None:
    """Translate one string, retrying transient failures. None if it never works."""
    last_err: str = "empty result"
    for attempt in range(retries):
        try:
            result = translator.translate(text)
            if result:
                return result
        except Exception as e:  # noqa: BLE001 - deep-translator raises many types
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    print(
        f"WARNING: translation request failed after {retries} attempts ({last_err}).",
        file=sys.stderr,
    )
    return None


def translate_text(
    text: str, target: str = "he", source: str = "en", retries: int = 3
) -> str:
    """Translate ``text`` from ``source`` to ``target``.

    Returns the translation, or the original ``text`` if deep-translator is
    missing or the service fails (a warning is printed).
    """
    if not text.strip():
        return text

    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print(
            "WARNING: deep-translator not installed - leaving text untranslated. "
            "Install it with: pip install deep-translator",
            file=sys.stderr,
        )
        return text

    tgt = _LANG_CODES.get(target, target)
    src = _LANG_CODES.get(source, source)

    try:
        translator = GoogleTranslator(source=src, target=tgt)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not init translator ({e}).", file=sys.stderr)
        return text

    if len(text) <= _MAX_SINGLE_CALL:
        whole = _translate_once(translator, text, retries)
        if whole is not None:
            return whole
        # fall through to paragraph mode as a second try

    # Paragraph-by-paragraph: preserves blank-line structure, smaller requests.
    out: list[str] = []
    for block in text.split("\n\n"):
        if not block.strip():
            out.append(block)
            continue
        piece = _translate_once(translator, block, retries)
        if piece is None:
            print(
                "WARNING: translation failed - returning the English text.",
                file=sys.stderr,
            )
            return text
        out.append(piece)
    return "\n\n".join(out)
