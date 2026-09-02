"""Main entry point for generating flower treatment instructions."""

from __future__ import annotations

from .data_loader import load_flower_data
from .llm_renderer import VALID_AUDIENCES
from .template_renderer import render_instructions_en, render_instructions_he

VALID_LANGUAGES = ("en", "he")
VALID_MODES = ("template", "llm")


def generate_treatment_instructions(
    flower_name: str,
    is_open: bool,
    yellow_leaves: bool = False,
    leaves_falling: bool = False,
    language: str = "en",
    mode: str = "template",
    audience: str = "farmer",
    model: str | None = None,
    translate: bool = True,
) -> str:
    """Generate shelf life extension instructions for a picked flower.

    Args:
        flower_name: English name of the flower (looked up in flowers.json).
        is_open: Water-entry handling. True renders the CLOSED-flower wording
            (controlled harvest); False renders the OPEN-flower wording (timing
            critical). Either way the 15-minute water-entry time is stated.
        yellow_leaves: Whether leaves are yellow (True = add Gibberellin).
        leaves_falling: Whether leaves are falling (reserved for future use).
        language: Output language - "en" for English, "he" for Hebrew.
        mode: Generation mode - "template" (default) or "llm" (uses Ollama).
        audience: Who the LLM writes for - "farmer" (default), "agronomist" or
            "layperson". Only affects mode="llm".
        model: Ollama model tag for mode="llm" (default: OLLAMA_MODEL env var,
            else llama3.2:latest).
        translate: For language="he" + mode="llm". True (default) composes in
            English and machine-translates to Hebrew; False trusts the model to
            write Hebrew directly.

    Returns:
        Formatted treatment instruction text.

    Raises:
        ValueError: If flower_name not found, or language/mode/audience is invalid.
    """
    # Validate inputs
    language = language.strip().lower()
    if language not in VALID_LANGUAGES:
        raise ValueError(
            f"Invalid language '{language}'. Must be one of: {', '.join(VALID_LANGUAGES)}"
        )

    mode = mode.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {', '.join(VALID_MODES)}"
        )

    audience = audience.strip().lower()
    if audience not in VALID_AUDIENCES:
        raise ValueError(
            f"Invalid audience '{audience}'. Must be one of: {', '.join(VALID_AUDIENCES)}"
        )

    # Load flower data (raises ValueError if not found)
    flower_data = load_flower_data(flower_name)

    # Generate instructions based on mode
    if mode == "llm":
        from .llm_renderer import render_instructions_llm

        return render_instructions_llm(
            flower_data=flower_data,
            is_open=is_open,
            yellow_leaves=yellow_leaves,
            leaves_falling=leaves_falling,
            language=language,
            audience=audience,
            model=model,
            translate=translate,
        )

    # Template mode
    if language == "en":
        return render_instructions_en(flower_data, is_open, yellow_leaves, leaves_falling)
    else:
        return render_instructions_he(flower_data, is_open, yellow_leaves, leaves_falling)
