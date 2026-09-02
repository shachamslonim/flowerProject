"""Template-based instruction renderer for flower shelf life treatment.

Generates structured instructions in English or Hebrew based on flower
preservation data from flowers.json.

Section order:
1. Sterilization
2. Water Entry (harvest to water introduction)
3. Materials and Chemical Treatment (biocides)
4. Problem-specific treatments (if applicable)
5. Environmental Conditions
6. Water Bucket Height (atmospheric pressure)

Materials purpose explanations:
- TOG products (TOG-6, TOG-10, TOG-30, TOG-75, TOG-Galileo) = Biocides
  (fungicides and bactericides for killing fungi and bacteria in water)
- STS (TOG-L-101) = Ethylene blocker (prevents petal drop due to ethylene sensitivity)
- Gibberellin = Prevents leaf yellowing
- 101L = Ethylene blocker
"""

from __future__ import annotations

from typing import Any

# Constants
STERILIZER_TOOLS = ["bins", "shears", "tables", "guillotine"]
TEMPERATURE_C = 20
MOISTURE_PERCENT = 30
WATER_ENTRY_MINUTES = 15
STS_SENSITIVITY_THRESHOLD = 3  # ethyleneSensitivity >= 3 triggers STS recommendation

# Product preference order (best first). Products not in this list are sorted last.
PRODUCT_PRIORITY_ORDER = ["TOG-6", "TOG-10", "TOG-30", "TOG-75", "TOG-Galileo", "TOG-3", "Sugar"]

# `priority` values from flowers.json, ranked best first. This is the primary
# sort key - the product order above is only a tie-breaker within a group.
PRIORITY_RANK = {"recommended": 0, "standard": 1}
PRIORITY_RANK_UNKNOWN = 2


def _parse_ethylene_sensitivity(value: Any) -> int | None:
    """Parse ethyleneSensitivity field which can be '1', '2', '3', '4', '1 (2)', or None."""
    if value is None:
        return None
    s = str(value).strip()
    # Handle cases like "1 (2)" - take the first digit
    if s and s[0].isdigit():
        return int(s[0])
    return None


def _is_recommended(treatment: dict[str, Any]) -> bool:
    """Whether a treatment is flagged as recommended in flowers.json."""
    return str(treatment.get("priority", "")).strip().lower() == "recommended"


def _sort_treatments(treatments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort treatments by the `priority` field from flowers.json.

    All 'recommended' products come first, then 'standard'. Within the same
    priority group, PRODUCT_PRIORITY_ORDER breaks the tie (TOG-6, TOG-10,
    TOG-30, TOG-75, ...). A flower may have more than one recommended product.
    """
    def sort_key(t: dict[str, Any]) -> tuple[int, int]:
        priority = str(t.get("priority", "")).strip().lower()
        priority_rank = PRIORITY_RANK.get(priority, PRIORITY_RANK_UNKNOWN)

        name = t.get("productName", "")
        try:
            product_rank = PRODUCT_PRIORITY_ORDER.index(name)
        except ValueError:
            product_rank = len(PRODUCT_PRIORITY_ORDER)  # unknown products go last

        return (priority_rank, product_rank)

    return sorted(treatments, key=sort_key)


def _get_rank_label_en(index: int, recommended: bool = False) -> str:
    """Get English rank label for a treatment by its sorted position.

    Args:
        index: Zero-based position in the sorted list.
        recommended: Whether flowers.json flags this product as recommended.
    """
    labels = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    label = labels[index] if index < len(labels) else f"Option {index + 1}"
    return f"{label}, recommended" if recommended else label


def _get_rank_label_he(index: int, recommended: bool = False) -> str:
    """Get Hebrew rank label for a treatment by its sorted position.

    Args:
        index: Zero-based position in the sorted list.
        recommended: Whether flowers.json flags this product as recommended.
    """
    labels = ["אפשרות 1", "אפשרות 2", "אפשרות 3", "אפשרות 4", "אפשרות 5"]
    label = labels[index] if index < len(labels) else f"אפשרות {index + 1}"
    return f"{label}, מומלץ" if recommended else label


def render_instructions_en(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
) -> str:
    """Render treatment instructions in English.

    Args:
        flower_data: Complete flower dict from flowers.json.
        is_open: Whether the flower is open (affects water entry time).
        yellow_leaves: Whether leaves are yellow (triggers Gibberellin).
        leaves_falling: Whether leaves are falling (reserved for future use).

    Returns:
        Formatted instruction text in English.
    """
    preservation = flower_data.get("preservation", {})
    english_name = flower_data.get("englishName", "Unknown")
    lines: list[str] = []

    # Header
    lines.append(f"=== Treatment Instructions for {english_name} ===")
    lines.append("")

    # Section 1: Sterilization
    lines.append("1. STERILIZATION")
    tools_str = ", ".join(STERILIZER_TOOLS)
    lines.append(f"   Sterilize the following tools with acetone: {tools_str}")
    lines.append("")

    # Section 2: Water Entry Time.
    # NOTE: is_open=True describes the CLOSED-flower handling and vice versa.
    lines.append("2. WATER ENTRY (HARVEST TO WATER INTRODUCTION)")
    if not is_open:
        lines.append(f"   Flower is OPEN: Water entry time is {WATER_ENTRY_MINUTES} minutes.")
        lines.append("   When flowers are open, timing between harvest and water")
        lines.append("   introduction is critical.")
    else:
        lines.append(f"   Flower is CLOSED: Water entry time is {WATER_ENTRY_MINUTES} minutes.")
        lines.append("   Controlled harvest. Harvest when flowers are closed.")
        lines.append("   Time between harvest and water introduction (sink) is important.")
    lines.append("")

    # Section 3: Materials / Chemical Treatment
    section_num = 3
    lines.append(f"{section_num}. MATERIALS AND CHEMICAL TREATMENT")
    lines.append("   Purpose: Biocides - fungicides and bactericides for killing fungi")
    lines.append("   and bacteria in the water solution.")
    lines.append("")
    farmer_treatments = preservation.get("farmerTreatment", [])
    if farmer_treatments:
        sorted_treatments = _sort_treatments(farmer_treatments)
        for i, treatment in enumerate(sorted_treatments):
            name = treatment.get("productName", "Unknown")
            concentration = treatment.get("concentrationRate", "N/A")
            method = treatment.get("applicationMethod", "N/A")
            rank = _get_rank_label_en(i, _is_recommended(treatment))
            line = f"   - {name} ({rank}): concentration {concentration}, method: {method}"
            if name == "Sugar":
                line += " [can be added to the option]"
            lines.append(line)
    else:
        lines.append("   No farmer treatment specified.")

    # Ethylene blocker: prefer the rate listed in the DOC table, otherwise fall
    # back to a generic STS recommendation driven by ethylene sensitivity.
    sensitivity = _parse_ethylene_sensitivity(preservation.get("ethyleneSensitivity"))
    blocker = preservation.get("ethyleneBlocker")
    if blocker:
        lines.append("")
        lines.append("   ETHYLENE BLOCKER:")
        suffix = ", recommended" if _is_recommended(blocker) else ""
        lines.append(
            f"   - STS ({blocker.get('productName')}{suffix}): "
            f"concentration {blocker.get('concentrationRate')}, "
            f"method: {blocker.get('applicationMethod')}"
        )
        lines.append("     Purpose: Prevents petal drop caused by ethylene sensitivity.")
    elif sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD:
        lines.append("")
        lines.append("   ETHYLENE BLOCKER:")
        lines.append("   - Use STS (TOG-L-101) - ethylene sensitivity detected")
        lines.append("     Purpose: Prevents petal drop caused by ethylene sensitivity.")

    # Leaf treatment (TOG-L-103) - guards against leaf yellowing
    leaf = preservation.get("leafTreatment")
    if leaf:
        lines.append("")
        lines.append("   LEAF TREATMENT:")
        suffix = ", recommended" if _is_recommended(leaf) else ""
        lines.append(
            f"   - {leaf.get('productName')}{suffix}: "
            f"concentration {leaf.get('concentrationRate')}, "
            f"method: {leaf.get('applicationMethod')}"
        )
        lines.append("     Purpose: Protects foliage and prevents leaf yellowing.")

    # Additives
    additives = preservation.get("additives")
    if additives:
        lines.append("")
        lines.append(f"   Additives: {additives}")

    lines.append("")

    # Section 4: Problem-specific treatments (if applicable)
    has_problems = yellow_leaves or leaves_falling
    if has_problems:
        section_num += 1
        lines.append(f"{section_num}. PROBLEM-SPECIFIC TREATMENTS")
        if yellow_leaves:
            lines.append("   - Yellow leaves: Apply Gibberellin")
            lines.append("     Purpose: Prevents and treats leaf yellowing.")
        if leaves_falling:
            lines.append("   - Leaves falling: Apply 101L")
            lines.append("     Purpose: Ethylene blocker - prevents leaf drop.")
        lines.append("")

    # Section: Environmental Conditions
    section_num += 1
    lines.append(f"{section_num}. ENVIRONMENTAL CONDITIONS")
    lines.append(f"   Temperature: {TEMPERATURE_C}\u00b0C")
    lines.append(f"   Moisture: {MOISTURE_PERCENT}%")
    lines.append("")

    # Section: Water Bucket Height
    section_num += 1
    water_height = preservation.get("waterBucketHeight")
    lines.append(f"{section_num}. WATER BUCKET HEIGHT (ATMOSPHERIC PRESSURE)")
    if water_height is not None:
        lines.append(f"   Water bucket height: {water_height} cm")
    else:
        lines.append("   Water bucket height: not specified")
    lines.append("")

    return "\n".join(lines)


# Hebrew tool names mapping
STERILIZER_TOOLS_HE = ["מיכלים", "מזמרות", "שולחנות", "גיליוטינה"]


def render_instructions_he(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool,
    leaves_falling: bool,
) -> str:
    """Render treatment instructions in Hebrew.

    Args:
        flower_data: Complete flower dict from flowers.json.
        is_open: Whether the flower is open (affects water entry time).
        yellow_leaves: Whether leaves are yellow (triggers Gibberellin).
        leaves_falling: Whether leaves are falling (reserved for future use).

    Returns:
        Formatted instruction text in Hebrew.
    """
    preservation = flower_data.get("preservation", {})
    hebrew_name = flower_data.get("hebrewName", "לא ידוע")
    lines: list[str] = []

    # Header
    lines.append(f"=== הוראות טיפול עבור {hebrew_name} ===")
    lines.append("")

    # Section 1: Sterilization
    lines.append("1. חיטוי")
    tools_str = ", ".join(STERILIZER_TOOLS_HE)
    lines.append(f"   יש לחטא את הכלים הבאים באצטון: {tools_str}")
    lines.append("")

    # Section 2: Water Entry Time (is_open=True describes the CLOSED handling)
    lines.append("2. כניסת מים (זמן בין קטיף להכנסה למים)")
    if not is_open:
        lines.append(f"   הפרח פתוח: זמן כניסת מים {WATER_ENTRY_MINUTES} דקות.")
        lines.append("   כשהפרחים פתוחים, הזמן בין הקטיף להכנסה למים קריטי.")
    else:
        lines.append(f"   הפרח סגור: זמן כניסת מים {WATER_ENTRY_MINUTES} דקות.")
        lines.append("   קטיף מבוקר. יש לקטוף כשהפרחים סגורים.")
        lines.append("   זמן בין קטיף להכנסה למים (sink) חשוב.")
    lines.append("")

    # Section 3: Materials / Chemical Treatment
    section_num = 3
    lines.append(f"{section_num}. חומרים וטיפול כימי")
    lines.append("   מטרה: ביוצידים – חומרי קטילת פטריות וחיידקים במי ההשקיה.")
    lines.append("")
    farmer_treatments = preservation.get("farmerTreatment", [])
    if farmer_treatments:
        sorted_treatments = _sort_treatments(farmer_treatments)
        for i, treatment in enumerate(sorted_treatments):
            name = treatment.get("productName", "לא ידוע")
            concentration = treatment.get("concentrationRate", "לא צוין")
            method = treatment.get("applicationMethod", "לא צוין")
            rank = _get_rank_label_he(i, _is_recommended(treatment))
            line = f"   - {name} ({rank}): ריכוז {concentration}, שיטת יישום: {method}"
            if name == "Sugar":
                line += " [ניתן להוסיף לאפשרות]"
            lines.append(line)
    else:
        lines.append("   לא צוין טיפול למגדל.")

    # Ethylene blocker: prefer the rate listed in the DOC table, otherwise fall
    # back to a generic STS recommendation driven by ethylene sensitivity.
    sensitivity = _parse_ethylene_sensitivity(preservation.get("ethyleneSensitivity"))
    blocker = preservation.get("ethyleneBlocker")
    if blocker:
        lines.append("")
        lines.append("   חוסם אתילן:")
        suffix = ", מומלץ" if _is_recommended(blocker) else ""
        lines.append(
            f"   - STS ({blocker.get('productName')}{suffix}): "
            f"ריכוז {blocker.get('concentrationRate')}, "
            f"שיטת יישום: {blocker.get('applicationMethod')}"
        )
        lines.append("     מטרה: מניעת נשירה הנגרמת מרגישות לאתילן.")
    elif sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD:
        lines.append("")
        lines.append("   חוסם אתילן:")
        lines.append("   - יש להשתמש ב-STS (TOG-L-101) - רגישות לאתילן")
        lines.append("     מטרה: מניעת נשירה הנגרמת מרגישות לאתילן.")

    # Leaf treatment (TOG-L-103) - guards against leaf yellowing
    leaf = preservation.get("leafTreatment")
    if leaf:
        lines.append("")
        lines.append("   טיפול בעלים:")
        suffix = ", מומלץ" if _is_recommended(leaf) else ""
        lines.append(
            f"   - {leaf.get('productName')}{suffix}: "
            f"ריכוז {leaf.get('concentrationRate')}, "
            f"שיטת יישום: {leaf.get('applicationMethod')}"
        )
        lines.append("     מטרה: הגנה על העלווה ומניעת הצהבת עלים.")

    # Additives
    additives = preservation.get("additives")
    if additives:
        lines.append("")
        lines.append(f"   תוספים: {additives}")

    lines.append("")

    # Section 4: Problem-specific treatments (if applicable)
    has_problems = yellow_leaves or leaves_falling
    if has_problems:
        section_num += 1
        lines.append(f"{section_num}. טיפולים לפי בעיות")
        if yellow_leaves:
            lines.append("   - הצהבת עלים: יש להשתמש בג'יברלין")
            lines.append("     מטרה: מניעת וטיפול בהצהבת עלים.")
        if leaves_falling:
            lines.append("   - נשירת עלים: יש להשתמש ב-101L")
            lines.append("     מטרה: חוסם אתילן – מניעת נשירת עלים.")
        lines.append("")

    # Section: Environmental Conditions
    section_num += 1
    lines.append(f"{section_num}. תנאי סביבה")
    lines.append(f"   טמפרטורה: {TEMPERATURE_C}\u00b0C")
    lines.append(f"   לחות: {MOISTURE_PERCENT}%")
    lines.append("")

    # Section: Water Bucket Height
    section_num += 1
    water_height = preservation.get("waterBucketHeight")
    lines.append(f"{section_num}. גובה מים בדלי (לחץ אטמוספרי)")
    if water_height is not None:
        lines.append(f'   גובה מים בדלי: {water_height} ס"מ')
    else:
        lines.append("   גובה מים בדלי: לא צוין")
    lines.append("")

    return "\n".join(lines)
