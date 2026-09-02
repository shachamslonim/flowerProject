"""Render treatment instructions as a styled HTML page."""

from __future__ import annotations

import html
from typing import Any

from .template_renderer import (
    MOISTURE_PERCENT,
    PRODUCT_PRIORITY_ORDER,
    STS_SENSITIVITY_THRESHOLD,
    STERILIZER_TOOLS,
    STERILIZER_TOOLS_HE,
    TEMPERATURE_C,
    WATER_ENTRY_MINUTES,
    _get_rank_label_en,
    _get_rank_label_he,
    _is_recommended,
    _parse_ethylene_sensitivity,
    _sort_treatments,
)

# ─── HTML boilerplate ────────────────────────────────────────────────────────

_CSS = """\
:root { --accent: #2e7d32; --bg: #fafafa; --card: #ffffff; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       background: var(--bg); color: #212121; padding: 2rem; line-height: 1.6; }
.container { max-width: 800px; margin: auto; }
h1 { color: var(--accent); margin-bottom: 1.5rem; border-bottom: 3px solid var(--accent);
     padding-bottom: .5rem; }
.section { background: var(--card); border-radius: 8px; padding: 1.25rem 1.5rem;
           margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.section h2 { font-size: 1.1rem; color: var(--accent); margin-bottom: .75rem;
              text-transform: uppercase; letter-spacing: .04em; }
.section p, .section li { margin-bottom: .4rem; }
ul { list-style: none; padding-left: 0; }
ul li::before { content: '\\2022'; color: var(--accent); font-weight: bold;
               display: inline-block; width: 1em; margin-left: -0em; }
.badge { display: inline-block; background: var(--accent); color: #fff;
         font-size: .75rem; padding: 2px 8px; border-radius: 12px; margin-left: .4rem; }
.badge.standard { background: #757575; }
.note { font-size: .85rem; color: #616161; font-style: italic; margin-left: .4rem; }
table.env { border-collapse: collapse; }
table.env td { padding: 4px 12px 4px 0; }
table.env td:first-child { font-weight: 600; }
.rtl { direction: rtl; text-align: right; }
"""

_HEAD = """\
<!DOCTYPE html>
<html lang="{lang}" {dir_attr}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
"""

_FOOT = """\
</div>
</body>
</html>
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


# ─── Public API ──────────────────────────────────────────────────────────────


def render_instructions_html(
    flower_data: dict[str, Any],
    is_open: bool,
    yellow_leaves: bool = False,
    leaves_falling: bool = False,
    language: str = "en",
) -> str:
    """Render treatment instructions as a complete HTML document.

    Args:
        flower_data: Single flower dict from flowers.json.
        is_open: Whether the flower is open.
        yellow_leaves: Whether to include Gibberellin section.
        leaves_falling: Whether to include 101L section.
        language: "en" or "he".

    Returns:
        Complete HTML string.
    """
    is_hebrew = language == "he"
    name = flower_data.get("hebrewName" if is_hebrew else "englishName", "Unknown")
    title = f"{'הוראות טיפול – ' if is_hebrew else 'Treatment Instructions – '}{name}"
    dir_attr = 'dir="rtl"' if is_hebrew else ""

    parts: list[str] = []
    parts.append(_HEAD.format(lang=language, dir_attr=dir_attr, title=_esc(title), css=_CSS))
    parts.append(f"<h1>{_esc(title)}</h1>\n")

    preservation = flower_data.get("preservation", {})

    # 1. Sterilization
    parts.append(_section_sterilization(is_hebrew))

    # 2. Water Entry
    parts.append(_section_water_entry(is_open, is_hebrew))

    # 3. Materials
    parts.append(_section_materials(preservation, is_hebrew))

    # 4. Problem-specific
    if yellow_leaves or leaves_falling:
        parts.append(_section_problems(yellow_leaves, leaves_falling, is_hebrew))

    # 5. Environment
    parts.append(_section_environment(is_hebrew))

    # 6. Water Bucket Height
    parts.append(_section_water_height(preservation, is_hebrew))

    parts.append(_FOOT)
    return "".join(parts)


# ─── Section builders ────────────────────────────────────────────────────────


def _section_sterilization(he: bool) -> str:
    heading = "חיטוי" if he else "Sterilization"
    tools = ", ".join(STERILIZER_TOOLS_HE if he else STERILIZER_TOOLS)
    body = (
        f"יש לחטא את הכלים הבאים באצטון: {tools}"
        if he
        else f"Sterilize the following tools with acetone: {tools}"
    )
    return f'<div class="section"><h2>{heading}</h2><p>{_esc(body)}</p></div>\n'


def _section_water_entry(is_open: bool, he: bool) -> str:
    heading = "כניסת מים" if he else "Water Entry"
    # is_open=True describes the CLOSED-flower handling and vice versa.
    if not is_open:
        if he:
            body = (
                f"<strong>הפרח פתוח:</strong> זמן כניסת מים {WATER_ENTRY_MINUTES} דקות."
                "<br>כשהפרחים פתוחים, הזמן בין הקטיף להכנסה למים קריטי."
            )
        else:
            body = (
                f"<strong>Flower is OPEN:</strong> Water entry time is {WATER_ENTRY_MINUTES} minutes."
                "<br>When flowers are open, timing between harvest and water introduction is critical."
            )
    else:
        if he:
            body = (
                f"<strong>הפרח סגור:</strong> זמן כניסת מים {WATER_ENTRY_MINUTES} דקות."
                "<br>קטיף מבוקר. יש לקטוף כשהפרחים סגורים."
                "<br>זמן בין קטיף להכנסה למים (sink) חשוב."
            )
        else:
            body = (
                f"<strong>Flower is CLOSED:</strong> Water entry time is {WATER_ENTRY_MINUTES} minutes."
                "<br>Controlled harvest. Harvest when flowers are closed."
                "<br>Time between harvest and water introduction (sink) is important."
            )
    return f'<div class="section"><h2>{heading}</h2><p>{body}</p></div>\n'


def _section_materials(preservation: dict, he: bool) -> str:
    heading = "חומרים וטיפול כימי" if he else "Materials and Chemical Treatment"
    purpose = (
        "ביוצידים – חומרי קטילת פטריות וחיידקים במי ההשקיה."
        if he
        else "Biocides – fungicides and bactericides for killing fungi and bacteria in the water solution."
    )

    lines: list[str] = [f'<div class="section"><h2>{heading}</h2>']
    lines.append(f"<p><em>{_esc(purpose)}</em></p>")

    farmer_treatments = preservation.get("farmerTreatment", [])
    if farmer_treatments:
        sorted_treatments = _sort_treatments(farmer_treatments)
        rank_fn = _get_rank_label_he if he else _get_rank_label_en
        lines.append("<ul>")
        for i, t in enumerate(sorted_treatments):
            name = t.get("productName", "Unknown")
            conc = t.get("concentrationRate", "N/A")
            method = t.get("applicationMethod", "N/A")
            recommended = _is_recommended(t)
            rank = rank_fn(i, recommended)
            badge = (
                '<span class="badge">מומלץ</span>'
                if he and recommended
                else '<span class="badge">recommended</span>'
                if recommended
                else '<span class="badge standard">standard</span>'
            )
            sugar_note = ""
            if name == "Sugar":
                sugar_note = (
                    ' <span class="note">[ניתן להוסיף לאפשרות]</span>'
                    if he
                    else ' <span class="note">[can be added to the option]</span>'
                )
            method_label = "שיטת יישום" if he else "method"
            conc_label = "ריכוז" if he else "concentration"
            lines.append(
                f"<li><strong>{_esc(name)}</strong> ({_esc(rank)}){badge}"
                f" — {conc_label}: {_esc(conc)}, {method_label}: {_esc(method)}"
                f"{sugar_note}</li>"
            )
        lines.append("</ul>")
    else:
        no_msg = "לא צוין טיפול למגדל." if he else "No farmer treatment specified."
        lines.append(f"<p>{no_msg}</p>")

    # Ethylene blocker
    sensitivity = _parse_ethylene_sensitivity(preservation.get("ethyleneSensitivity"))
    blocker = preservation.get("ethyleneBlocker")
    if blocker:
        sub = "חוסם אתילן" if he else "Ethylene Blocker"
        recommended = _is_recommended(blocker)
        badge = (
            '<span class="badge">מומלץ</span>'
            if he and recommended
            else '<span class="badge">recommended</span>'
            if recommended
            else ""
        )
        conc = blocker.get("concentrationRate", "N/A")
        method = blocker.get("applicationMethod", "N/A")
        lines.append(f"<h3 style='margin-top:1rem;color:#c62828'>{sub}</h3>")
        lines.append(
            f"<p>STS ({_esc(blocker.get('productName', 'TOG-L-101'))}) {badge}"
            f" — {'ריכוז' if he else 'concentration'}: {_esc(conc)}, "
            f"{'שיטת יישום' if he else 'method'}: {_esc(method)}</p>"
        )
    elif sensitivity is not None and sensitivity >= STS_SENSITIVITY_THRESHOLD:
        sub = "חוסם אתילן" if he else "Ethylene Blocker"
        lines.append(f"<h3 style='margin-top:1rem;color:#c62828'>{sub}</h3>")
        msg = (
            "יש להשתמש ב-STS (TOG-L-101) – רגישות לאתילן"
            if he
            else "Use STS (TOG-L-101) – ethylene sensitivity detected"
        )
        lines.append(f"<p>{msg}</p>")

    # Leaf treatment
    leaf = preservation.get("leafTreatment")
    if leaf:
        sub = "טיפול בעלים" if he else "Leaf Treatment"
        recommended = _is_recommended(leaf)
        badge = (
            '<span class="badge">מומלץ</span>'
            if he and recommended
            else '<span class="badge">recommended</span>'
            if recommended
            else ""
        )
        lines.append(f"<h3 style='margin-top:1rem;color:#1b5e20'>{sub}</h3>")
        lines.append(
            f"<p>{_esc(leaf.get('productName', ''))} {badge}"
            f" — {'ריכוז' if he else 'concentration'}: {_esc(leaf.get('concentrationRate', ''))}, "
            f"{'שיטת יישום' if he else 'method'}: {_esc(leaf.get('applicationMethod', ''))}</p>"
        )

    # Additives
    additives = preservation.get("additives")
    if additives:
        sub = "תוספים" if he else "Additives"
        lines.append(f"<p style='margin-top:.75rem'><strong>{sub}:</strong> {_esc(additives)}</p>")

    lines.append("</div>\n")
    return "".join(lines)


def _section_problems(yellow: bool, falling: bool, he: bool) -> str:
    heading = "טיפולים בעייתיים" if he else "Problem-Specific Treatments"
    lines = [f'<div class="section"><h2>{heading}</h2><ul>']
    if yellow:
        if he:
            lines.append("<li><strong>עלים צהובים:</strong> יש ליישם ג'יברלין – מניעת הצהבת עלים.</li>")
        else:
            lines.append("<li><strong>Yellow leaves:</strong> Apply Gibberellin – prevents leaf yellowing.</li>")
    if falling:
        if he:
            lines.append("<li><strong>נשירת עלים:</strong> יש להשתמש ב-101L – חוסם אתילן.</li>")
        else:
            lines.append("<li><strong>Leaves falling:</strong> Apply 101L – ethylene blocker.</li>")
    lines.append("</ul></div>\n")
    return "".join(lines)


def _section_environment(he: bool) -> str:
    heading = "תנאי סביבה" if he else "Environmental Conditions"
    temp_label = "טמפרטורה" if he else "Temperature"
    moist_label = "לחות" if he else "Moisture"
    return (
        f'<div class="section"><h2>{heading}</h2>'
        f'<table class="env">'
        f"<tr><td>{temp_label}</td><td>{TEMPERATURE_C}°C</td></tr>"
        f"<tr><td>{moist_label}</td><td>{MOISTURE_PERCENT}%</td></tr>"
        f"</table></div>\n"
    )


def _section_water_height(preservation: dict, he: bool) -> str:
    heading = "גובה מים בדלי" if he else "Water Bucket Height"
    height = preservation.get("waterBucketHeight")
    if height is not None:
        value = f"{height} {'ס\"מ' if he else 'cm'}"
    else:
        value = "לא צוין" if he else "not specified"
    return f'<div class="section"><h2>{heading}</h2><p>{value}</p></div>\n'
