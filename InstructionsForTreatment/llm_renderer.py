"""LLM-based instruction renderer using Ollama (llama3.2).

Rephrases the same structured data into natural language instructions
using a local LLM via langchain_ollama.
"""

from __future__ import annotations

import sys
from typing import Any

from .template_renderer import (
    MOISTURE_PERCENT,
    STERILIZER_TOOLS,
    STS_SENSITIVITY_THRESHOLD,
    TEMPERATURE_C,
    WATER_ENTRY_MINUTES,
    _parse_ethylene_sensitivity,
)

OLLAMA_MODEL = "llama3.2:latest"

SYSTEM_PROMPT_EN = """\
You are a professional agricultural advisor writing treatment instructions for extending \
the shelf life of picked flowers. You will receive structured data about a specific flower \
and must produce clear, professional instructions for growers.

Rules:
- Use ONLY the data provided. Do not invent or add information.
- Follow this exact section order: Sterilization → Materials/Chemicals → Leaf Treatment (if applicable) → Environmental Conditions → Water Bucket → Water Entry (if applicable).
- Write in English.
- Be concise and professional.
- Format with numbered sections and clear bullet points.
"""

SYSTEM_PROMPT_HE = """\
אתה יועץ חקלאי מקצועי הכותב הוראות טיפול להארכת חיי מדף של פרחים קטופים. תקבל מידע \
מובנה על פרח מסוים ועליך לייצר הוראות ברורות ומקצועיות למגדלים.

כללים:
- השתמש אך ורק בנתונים שסופקו. אל תמציא או תוסיף מידע.
- עקוב אחר סדר הסעיפים הבא: חיטוי → חומרים/כימיקלים → טיפול בעלים (אם רלוונטי) → תנאי סביבה → גובה מים בדלי → כניסת מים (אם רלוונטי).
- כתוב בעברית.
- היה תמציתי ומקצועי.
- עצב עם סעיפים ממוספרים ונקודות ברורות.
"""


def _build_data_prompt(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
    language: str,
) -> str:
    """Build the user message containing all structured data for the LLM."""
    preservation = flower_data.get("preservation", {})
    name = flower_data.get("englishName", "Unknown")
    hebrew_name = flower_data.get("hebrewName", "לא ידוע")

    # Parse sensitivity
    sensitivity = _parse_ethylene_sensitivity(preservation.get("ethyleneSensitivity"))

    # Build data sections
    sections = []

    # Flower identity
    if language == "he":
        sections.append(f"שם הפרח: {hebrew_name} ({name})")
    else:
        sections.append(f"Flower: {name}")

    # Sterilization
    tools_str = ", ".join(STERILIZER_TOOLS)
    sections.append(f"Sterilization tools (clean with acetone): {tools_str}")

    # Materials
    farmer_treatments = preservation.get("farmerTreatment", [])
    if farmer_treatments:
        materials_lines = ["Materials/Chemical treatments:"]
        for t in farmer_treatments:
            materials_lines.append(
                f"  - {t.get('productName', '?')}: "
                f"concentration {t.get('concentrationRate', '?')}, "
                f"method: {t.get('applicationMethod', '?')}"
            )
        sections.append("\n".join(materials_lines))
    else:
        sections.append("Materials: No farmer treatment specified.")

    # STS
    if sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD:
        sections.append("IMPORTANT: Use STS (TOG-L-101) due to high ethylene sensitivity.")

    # Additives
    additives = preservation.get("additives")
    if additives:
        sections.append(f"Additives: {additives}")

    # Gibberellin
    if yellow_leaves:
        sections.append("Yellow leaves detected: Apply Gibberellin to prevent leaf yellowing.")

    # Environment
    sections.append(f"Temperature: {TEMPERATURE_C}°C")
    sections.append(f"Moisture: {MOISTURE_PERCENT}%")

    # Water bucket height
    water_height = preservation.get("waterBucketHeight")
    if water_height is not None:
        sections.append(f"Water bucket height (atmospheric pressure): {water_height} cm")
    else:
        sections.append("Water bucket height: not specified for this flower")

    # Water entry
    if is_open:
        sections.append(f"Flower is OPEN: Water entry time = {WATER_ENTRY_MINUTES} minutes")
    else:
        sections.append("Flower is CLOSED: No special water entry timing required.")

    return "\n".join(sections)


def render_instructions_llm(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
    language: str = "en",
) -> str:
    """Render treatment instructions using LLM (Ollama llama3.2).

    Falls back to template mode if Ollama is not available.

    Args:
        flower_data: Complete flower dict from flowers.json.
        is_open: Whether the flower is open.
        yellow_leaves: Whether leaves are yellow.
        leaves_falling: Whether leaves are falling.
        language: "en" for English, "he" for Hebrew.

    Returns:
        Natural language treatment instructions.
    """
    # Build the prompts
    system_prompt = SYSTEM_PROMPT_HE if language == "he" else SYSTEM_PROMPT_EN
    user_message = _build_data_prompt(flower_data, is_open, yellow_leaves, leaves_falling, language)

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        return response.content

    except ImportError:
        print(
            "WARNING: langchain_ollama not installed. Falling back to template mode.",
            file=sys.stderr,
        )
        return _fallback_to_template(flower_data, is_open, yellow_leaves, leaves_falling, language)

    except Exception as e:
        print(
            f"WARNING: Could not connect to Ollama ({e}). Falling back to template mode.",
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
