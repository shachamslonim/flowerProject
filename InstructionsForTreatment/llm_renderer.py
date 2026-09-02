"""LLM-based instruction renderer using a local Ollama model.

Rephrases the same structured data into natural language instructions
using a local LLM via langchain_ollama. The model is configurable
(see :func:`resolve_model`); the default ``llama3.2`` is English-only.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .template_renderer import (
    MOISTURE_PERCENT,
    STERILIZER_TOOLS,
    STS_SENSITIVITY_THRESHOLD,
    TEMPERATURE_C,
    WATER_ENTRY_MINUTES,
    _is_recommended,
    _parse_ethylene_sensitivity,
    _sort_treatments,
)

OLLAMA_MODEL = "llama3.2:latest"


def resolve_model(model: str | None = None) -> str:
    """Which Ollama model to use: explicit arg > OLLAMA_MODEL env > default.

    ``llama3.2`` cannot write coherent Hebrew (it loops); for ``language="he"``
    point this at a multilingual model, e.g. ``aya-expanse:8b`` or ``gemma2:9b``.
    """
    return model or os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL

# The reader the instructions are written for. Selected via --audience / the
# `audience` parameter; only affects LLM mode.
VALID_AUDIENCES = ("farmer", "agronomist", "layperson")

_DEFAULT_SYSTEM_BASE_EN = """\
You are turning structured data about one picked flower into shelf-life treatment \
instructions for ONE reader (described below).

The input has two kinds of content:
1. FIXED VALUES - product names, concentration rates, temperatures, percentages, \
heights and times. These are facts from a database. Copy EVERY one into your output \
exactly as written - same number, same product name. You may reorder them, but nothing \
may be dropped, added, merged, rounded, converted, or turned into a range.
2. Everything else is yours to phrase. In your own plain words, explain WHAT the reader \
does with each value and WHY.

Rules:
- Reproduce every FIXED VALUE verbatim - every product name, rate, and its \
(recommended/standard) mark.
- The biocide options are ALTERNATIVES: list them all so the reader can choose, each \
with its rate and mark, but make clear the reader picks only ONE (a "recommended" one \
if possible). The ethylene blocker, leaf treatment, additives and Sugar are add-ons \
used together with the chosen biocide, not choices.
- Never invent a value, product, step, range, or unit that is not in the data.
- Follow this section order: Sterilization -> Materials/Chemicals -> Leaf Treatment \
(if any) -> Environmental Conditions -> Water Bucket -> Water Entry.
- Write in English, with numbered sections and clear bullet points.
"""

# Directory where prompt_tuning writes optimized prompt overrides.
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt_override(name: str) -> str | None:
    """Return the contents of ``prompts/<name>.txt`` if it exists, else None.

    Lets the prompt-tuning loop (``tune_prompt.py``) swap in an improved prompt
    without editing this module. Delete the file to revert to the default.
    """
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except OSError:
        return None


def system_base_en() -> str:
    """English base prompt: the optimized override if present, else the default."""
    return load_prompt_override("system_en") or _DEFAULT_SYSTEM_BASE_EN


# Back-compat alias; prefer system_base_en() so a written override is picked up.
_SYSTEM_BASE_EN = _DEFAULT_SYSTEM_BASE_EN

_SYSTEM_BASE_HE = """\
אתה כותב הוראות טיפול להארכת חיי מדף של פרחים קטופים. תקבל מידע מובנה על פרח מסוים \
ועליך להפוך אותו להוראות ברורות עבור קורא אחד (המתואר בהמשך).

כללים:
- השתמש אך ורק בנתונים שסופקו. אל תמציא או תוסיף מידע.
- עקוב אחר סדר הסעיפים הבא: חיטוי → חומרים/כימיקלים → טיפול בעלים (אם רלוונטי) → תנאי סביבה → גובה מים בדלי → כניסת מים (אם רלוונטי).
- כתוב בעברית.
- עצב עם סעיפים ממוספרים ונקודות ברורות.
"""

AUDIENCE_GUIDANCE_EN = {
    "farmer": """\
Reader: a single working flower grower, out in the field.
- Write short, direct imperative steps.
- Use plain, practical field language.
- Keep the product names, concentrations and application methods, but skip the
  chemistry theory - focus on what to do and in what order.
- Be concise and professional.""",
    "agronomist": """\
Reader: a single crop specialist / agronomist.
- Use precise technical terminology.
- Include every concentration rate and application method from the data.
- Briefly state the reason for each treatment (mode of action: biocide,
  ethylene blocker, leaf protection, etc.).
- Professional, technical register.""",
    "layperson": """\
Reader: a single interested person with no agricultural background.
- Use everyday words.
- The first time a piece of jargon or a product code appears, explain in a few
  words what it is and what it does.
- Friendly, reassuring tone.
- Still cover every step - do not skip anything.""",
}

AUDIENCE_GUIDANCE_HE = {
    "farmer": """\
הקורא: מגדל פרחים אחד שעובד בשטח.
- כתוב שלבים קצרים וישירים בלשון ציווי.
- השתמש בשפת שטח פשוטה ומעשית.
- שמור על שמות המוצרים, הריכוזים ושיטות היישום, אך דלג על תיאוריית הכימיה -
  התמקד במה לעשות ובאיזה סדר.
- היה תמציתי ומקצועי.""",
    "agronomist": """\
הקורא: אגרונום / מומחה גידול אחד.
- השתמש במינוח טכני מדויק.
- כלול כל ריכוז וכל שיטת יישום מהנתונים.
- ציין בקצרה את מטרת כל טיפול (אופן הפעולה: ביוציד, חוסם אתילן, הגנה על עלים וכו').
- משלב מקצועי וטכני.""",
    "layperson": """\
הקורא: אדם אחד מתעניין ללא רקע חקלאי.
- השתמש במילים יומיומיות.
- בפעם הראשונה שמופיע מונח מקצועי או קוד מוצר, הסבר במילים ספורות מה זה ומה תפקידו.
- טון ידידותי ומרגיע.
- עדיין כסה כל שלב - אל תדלג על דבר.""",
}


def _build_system_prompt(language: str, audience: str) -> str:
    """Build the system prompt: base rules + guidance for the chosen reader."""
    base = _SYSTEM_BASE_HE if language == "he" else system_base_en()
    guidance = AUDIENCE_GUIDANCE_HE if language == "he" else AUDIENCE_GUIDANCE_EN
    return base + "\n" + guidance.get(audience, guidance["farmer"])


def _build_data_prompt(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
    language: str,
) -> str:
    """Build the user message: the FIXED VALUES block plus the task line.

    Everything under "FIXED VALUES" must survive into the LLM output verbatim; the
    system prompt tells the model it may only reword the text around them.
    """
    preservation = flower_data.get("preservation", {})
    name = flower_data.get("englishName", "Unknown")
    hebrew_name = flower_data.get("hebrewName", "לא ידוע")

    sensitivity = _parse_ethylene_sensitivity(preservation.get("ethyleneSensitivity"))

    ident = f"{hebrew_name} ({name})" if language == "he" else name

    # Each entry is a ready-made instruction fragment: the model must keep every
    # number and product name in it, and may rephrase the surrounding words.
    fixed: list[str] = []

    # Sterilization
    fixed.append(
        f"Clean these tools with acetone: {', '.join(STERILIZER_TOOLS)}."
    )

    # Materials - the biocides are RANKED ALTERNATIVES: the grower picks ONE
    # (prefer a "recommended" one). Sugar is an add-on, not an alternative.
    farmer_treatments = _sort_treatments(preservation.get("farmerTreatment", []))
    options = [t for t in farmer_treatments if t.get("productName") != "Sugar"]
    sugar = [t for t in farmer_treatments if t.get("productName") == "Sugar"]
    if options:
        opts = "; ".join(
            f"{t.get('productName', '?')} at {t.get('concentrationRate', '?')} "
            f"({'recommended' if _is_recommended(t) else 'standard'})"
            for t in options
        )
        fixed.append(
            f"Biocide for the water solution - list ALL {len(options)} of these options "
            "in your output (each with its rate and its recommended/standard mark), then "
            f"tell the grower to use only ONE of them (prefer a recommended one): {opts}."
        )
    else:
        fixed.append("No biocide options are listed for this flower.")
    for t in sugar:
        fixed.append(
            f"Sugar at {t.get('concentrationRate', '?')} can be added on top of the "
            "chosen biocide option (it is an add-on, not one of the choices)."
        )

    # Ethylene blocker (added alongside the chosen biocide, not an alternative)
    blocker = preservation.get("ethyleneBlocker")
    sensitivity_stated = False
    if blocker:
        mark = "recommended" if _is_recommended(blocker) else "standard"
        fixed.append(
            f"Also add the ethylene blocker STS ({blocker.get('productName', 'TOG-L-101')}) "
            f"at {blocker.get('concentrationRate', '?')} ({mark})."
        )
    elif sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD:
        fixed.append(
            f"This flower's ethylene sensitivity is {sensitivity}, so also add the "
            "ethylene blocker STS (TOG-L-101)."
        )
        sensitivity_stated = True

    # Leaf treatment (added alongside, not an alternative)
    leaf = preservation.get("leafTreatment")
    if leaf:
        mark = "recommended" if _is_recommended(leaf) else "standard"
        fixed.append(
            f"For the leaves, also use {leaf.get('productName', '?')} at "
            f"{leaf.get('concentrationRate', '?')} ({mark})."
        )

    # Additives
    additives = preservation.get("additives")
    if additives:
        fixed.append(f"Add the additive: {additives}.")

    # Raw ethylene sensitivity value (verbatim), unless already stated above
    raw_sens = preservation.get("ethyleneSensitivity")
    if raw_sens and not sensitivity_stated:
        fixed.append(f"This flower's ethylene sensitivity rating is {raw_sens}.")

    # Say plainly what is NOT needed, so the model does not invent these sections.
    not_needed = []
    if not blocker and not (sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD):
        not_needed.append("no ethylene blocker")
    if not leaf:
        not_needed.append("no leaf treatment")
    if not additives and not sugar:
        not_needed.append("no additives")
    if not_needed:
        fixed.append(f"This flower needs {', '.join(not_needed)}.")

    # Problem-specific
    if yellow_leaves:
        fixed.append("Yellow leaves were reported, so apply Gibberellin to stop the yellowing.")
    if leaves_falling:
        fixed.append("Falling leaves were reported, so apply 101L to block ethylene.")

    # Environment (fixed constants)
    fixed.append(f"Hold the flowers at {TEMPERATURE_C}°C and {MOISTURE_PERCENT}% humidity.")

    # Water bucket height
    water_height = preservation.get("waterBucketHeight")
    if water_height is not None:
        fixed.append(
            f"Fill the water bucket to {water_height} cm (this sets the atmospheric pressure)."
        )
    else:
        fixed.append("The water bucket height is not specified for this flower.")

    # Water entry - the template always states the 15-minute time.
    # NOTE: is_open=True describes the CLOSED-flower handling and vice versa.
    if not is_open:
        fixed.append(
            f"Get the flowers into water within {WATER_ENTRY_MINUTES} minutes of harvest. "
            "The flower is open, so this timing is critical."
        )
    else:
        fixed.append(
            f"Get the flowers into water within {WATER_ENTRY_MINUTES} minutes of harvest. "
            "The flower is closed, so harvest under control; the harvest-to-water time "
            "still matters."
        )

    fixed_block = "\n".join(f"- {line}" for line in fixed)
    return (
        f"Flower: {ident}\n\n"
        "=== FIXED VALUES ===\n"
        "Each line below is a fact. In your instructions you may rephrase the wording, "
        "merge lines, change their order, and add plain-language explanation of why - "
        "but keep every number and every product name exactly, drop nothing, and never "
        "write the words \"FIXED VALUES\" or copy a line as a bare label.\n"
        f"{fixed_block}\n\n"
        "=== YOUR JOB ===\n"
        "Write the treatment instructions for the reader described in the system message."
    )


# A complete instruction set runs several hundred chars over many lines; the
# stub-and-stop failure produces a handful of short lines.
_MIN_LLM_CHARS = 350
_MIN_LLM_LINES = 5


def _looks_complete(text: str) -> bool:
    return len(text) >= _MIN_LLM_CHARS and text.count("\n") >= _MIN_LLM_LINES


def render_instructions_llm(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
    language: str = "en",
    audience: str = "farmer",
    model: str | None = None,
    translate: bool = True,
) -> str:
    """Render treatment instructions using a local Ollama model.

    Falls back to template mode if Ollama is not available.

    Args:
        flower_data: Complete flower dict from flowers.json.
        is_open: Whether the flower is open.
        yellow_leaves: Whether leaves are yellow.
        leaves_falling: Whether leaves are falling.
        language: "en" for English, "he" for Hebrew.
        audience: Who to write for - "farmer", "agronomist" or "layperson".
        model: Ollama model tag. Defaults to the OLLAMA_MODEL env var, then
            ``llama3.2:latest``.
        translate: For ``language="he"`` only. When True (default), the model
            writes in English (where it is accurate) and the result is machine-
            translated to Hebrew. Set False if ``model`` genuinely writes Hebrew.

    Returns:
        Natural language treatment instructions.
    """
    # Small local models write accurate English but broken Hebrew, so compose in
    # English and translate afterwards unless the caller opted out.
    translate_after = translate and language == "he"
    compose_language = "en" if translate_after else language

    system_prompt = _build_system_prompt(compose_language, audience)
    user_message = _build_data_prompt(
        flower_data, is_open, yellow_leaves, leaves_falling, compose_language
    )
    resolved_model = resolve_model(model)

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        # repeat_penalty / num_predict guard against the runaway-repetition
        # failure mode small models fall into (badly on Hebrew).
        llm = ChatOllama(
            model=resolved_model,
            temperature=0.3,
            repeat_penalty=1.3,
            num_predict=900,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        # Small models sometimes emit a 2-3 line stub and stop. Retry once; if it
        # is still truncated, fall back to the (always-complete) template.
        text = ""
        for attempt in range(2):
            content = llm.invoke(messages).content
            text = (content if isinstance(content, str) else str(content)).strip()
            if _looks_complete(text):
                break
            print(
                f"WARNING: LLM output looks truncated ({len(text)} chars), "
                f"attempt {attempt + 1}/2.",
                file=sys.stderr,
            )
        else:
            print("WARNING: LLM kept truncating. Falling back to template.", file=sys.stderr)
            return _fallback_to_template(
                flower_data, is_open, yellow_leaves, leaves_falling, language
            )

        if translate_after:
            from .translate import translate_text

            print(
                f"Translating LLM output to Hebrew (model '{resolved_model}' "
                "wrote English)...",
                file=sys.stderr,
            )
            text = translate_text(text, target="he", source="en")

        return text

    except ImportError:
        print(
            "WARNING: langchain_ollama not installed. Falling back to template mode.",
            file=sys.stderr,
        )
        return _fallback_to_template(flower_data, is_open, yellow_leaves, leaves_falling, language)

    except Exception as e:
        print(
            f"WARNING: Ollama model '{resolved_model}' failed ({e}). "
            "Falling back to template mode.",
            file=sys.stderr,
        )
        return _fallback_to_template(flower_data, is_open, yellow_leaves, leaves_falling, language)


def _fallback_to_template(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
    language: str,
) -> str:
    """Fall back to template-based rendering."""
    from .template_renderer import render_instructions_en, render_instructions_he

    if language == "he":
        return render_instructions_he(flower_data, is_open, yellow_leaves, leaves_falling)
    return render_instructions_en(flower_data, is_open, yellow_leaves, leaves_falling)
