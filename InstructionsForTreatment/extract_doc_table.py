"""Extract the TOG recommended-use table from A_monthly_updated_match_schedule.doc.

Requires Microsoft Word (uses COM automation) because the source is a legacy
binary .doc and the "recommended" signal is carried by *font colour*: the DOC
legend states "Products & rates marked in red are highly adaptable."

Output: doc_table.json next to this file - a list of rows, each a mapping of
column name -> {"text": str, "red": bool}. sync_from_doc.py consumes it, so
Word is only needed when the DOC itself changes.

Usage (from the repo root, with the venv active):
    python -m InstructionsForTreatment.extract_doc_table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Column layout of the table, left to right. The header's first cell is
# vertically merged, so "Flower" is implicit in column 1.
COLUMNS = [
    "Flower",
    "Ethylene",
    "TOG-3",
    "TOG-6",
    "TOG-10",
    "TOG-30",
    "TOG-75",
    "TOG-Galileo",
    "TOG-L-101",
    "TOG-L-103",
    "Sugar",
    "Long Life SL",
    "Long life Bulb SG",
    "Long Life SG",
    "Additives",
]

WD_UNDEFINED = 9999999  # Word returns this when a range mixes colours


def _is_reddish(bgr: int) -> bool:
    """Whether a Word colour value is red-dominant.

    Word exposes colours as BGR integers. Negative values are theme/automatic
    colours (e.g. wdColorAutomatic) and are never red.
    """
    if bgr < 0 or bgr == WD_UNDEFINED:
        return False
    blue = (bgr >> 16) & 0xFF
    green = (bgr >> 8) & 0xFF
    red = bgr & 0xFF
    return red >= 128 and red > green + 60 and red > blue + 60


def _cell_text(raw: str) -> str:
    """Clean Word cell text (strips cell/row marks and non-breaking spaces)."""
    return (
        raw.replace("\r\x07", "")
        .replace("\x07", "")
        .replace("\xa0", " ")
        .replace("\r", " ")
        .replace("\x0b", " ")
        .strip()
    )


def extract(doc_path: Path | str | None = None, out_path: Path | str | None = None) -> int:
    """Extract table rows (with red flags) from the DOC into JSON.

    Returns:
        Number of rows written.
    """
    import win32com.client  # imported lazily so the module imports without Word

    root = Path(__file__).resolve().parent
    doc_path = Path(doc_path) if doc_path else root.parent / "A_monthly_updated_match_schedule.doc"
    out_path = Path(out_path) if out_path else root / "doc_table.json"

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False

    rows: list[dict[str, Any]] = []
    try:
        word.Documents.Open(str(doc_path))
        doc = word.ActiveDocument
        try:
            table = doc.Tables(1)
            for r in range(1, table.Rows.Count + 1):
                row: dict[str, Any] = {}
                for cell in table.Rows(r).Cells:
                    rng = cell.Range
                    colour = rng.Font.Color
                    red = _is_reddish(colour)
                    if colour == WD_UNDEFINED:
                        # Mixed formatting - red if any word in the cell is red
                        red = any(
                            w.Text.strip() and _is_reddish(w.Font.Color) for w in rng.Words
                        )
                    index = cell.ColumnIndex - 1
                    label = COLUMNS[index] if index < len(COLUMNS) else f"extra{index}"
                    row[label] = {"text": _cell_text(rng.Text), "red": red}
                rows.append(row)
        finally:
            doc.Close(False)
    finally:
        word.Quit()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return len(rows)


if __name__ == "__main__":
    count = extract()
    print(f"Extracted {count} rows to doc_table.json")
