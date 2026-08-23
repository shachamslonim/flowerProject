"""Sync flowers.json preservation data with the TOG recommended-use table.

Source of truth is ``doc_table.json``, produced by ``extract_doc_table.py``
from ``A_monthly_updated_match_schedule.doc``. That extraction preserves the
DOC's font colour, which is what encodes the recommendation:

    "Products & rates marked in red are highly adaptable."  -> priority=recommended
    "* = Dipping"                                           -> applicationMethod=dipping

What gets synced, per flower present in the DOC table:
  - ``ethyleneSensitivity`` (+ the matching scale description)
  - ``farmerTreatment``  - the Pre-treatment grower TOG columns, plus Sugar
  - ``ethyleneBlocker``  - TOG-L-101 (STS) with its actual rate, when listed
  - ``leafTreatment``    - TOG-L-103 with its actual rate, when listed
  - ``additives``        - the DOC "Additives" column

Products absent from a flower's DOC row are removed, so stale entries (for
example a TOG-10 rate that the table never listed) do not survive.

Flowers with no row in the DOC table are left untouched.
Idempotent - safe to run repeatedly.

Usage (from the repo root):
    python -m InstructionsForTreatment.sync_from_doc
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Pre-treatment grower columns, plus Sugar which the grower adds to the mix.
FARMER_COLUMNS = [
    "TOG-3",
    "TOG-6",
    "TOG-10",
    "TOG-30",
    "TOG-75",
    "TOG-Galileo",
    "Sugar",
]

ETHYLENE_BLOCKER_COLUMN = "TOG-L-101"
LEAF_TREATMENT_COLUMN = "TOG-L-103"

SENSITIVITY_DESCRIPTIONS = {
    1: "non sensitive",
    2: "low sensitivity",
    3: "moderate sensitivity",
    4: "very high sensitive",
}
SENSITIVITY_SCALE = "1=non sensitive, 2=low sensitivity, 3=moderate sensitivity, 4=very high sensitive"

# DOC row label -> flowers.json englishName, for names that differ.
NAME_ALIASES = {
    "roses": "Rose",
    "dianthus standard\\spray": "Dianthus standard/spray",
    "roscus": "Ruscus",
    "helianthus": "Sunflower",
    "lipidium": "Lepidium",
    "limonium sinutatum": "Statice/Limonium",
    "calla lily": "Lily",
}


def _normalize(name: str) -> str:
    """Lowercase and strip punctuation so DOC and JSON names can be compared."""
    name = re.sub(r"[^a-z0-9]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def _clean_row_label(raw: str) -> str:
    """Clean a DOC flower-name cell.

    The Grevillea cell carries a stale embedded copy of its own row, which
    begins at the first digit, so everything from there on is dropped.
    """
    label = re.sub(r"\s+", " ", raw).strip()
    trimmed = re.split(r"\d", label)[0].strip()
    return trimmed or label


def _parse_rate(text: str) -> tuple[str, str]:
    """Split a DOC rate cell into (rate, applicationMethod).

    A '*' anywhere in the cell means dipping, per the DOC legend.
    """
    method = "dipping" if "*" in text else "mixing"
    rate = text.replace("*", "").strip()
    return rate, method


def _treatment(product: str, cell: dict[str, Any]) -> dict[str, Any]:
    rate, method = _parse_rate(cell["text"])
    return {
        "productName": product,
        "concentrationRate": rate,
        "applicationMethod": method,
        "priority": "recommended" if cell.get("red") else "standard",
    }


def _load_doc_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load doc_table.json keyed by normalized flower name."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = (row.get("Flower") or {}).get("text", "")
        label = _clean_row_label(raw)
        if not label or label == "Flower" or label.startswith("Products &"):
            continue
        table[_normalize(label)] = row
    return table


def _lookup(english_name: str, table: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Find the DOC row for a flowers.json englishName."""
    key = _normalize(english_name)
    if key in table:
        return table[key]
    for doc_key, target in NAME_ALIASES.items():
        if target == english_name:
            row = table.get(_normalize(doc_key))
            if row:
                return row
    return None


def sync(
    json_path: Path | str | None = None,
    doc_table_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Rewrite preservation data for every flower found in the DOC table.

    Returns:
        (number of flowers changed, list of flowers with no DOC row)
    """
    here = Path(__file__).resolve().parent
    json_path = Path(json_path) if json_path else here.parent / "flowers.json"
    doc_table_path = Path(doc_table_path) if doc_table_path else here / "doc_table.json"

    table = _load_doc_rows(doc_table_path)

    with open(json_path, encoding="utf-8") as f:
        flowers = json.load(f)

    changed = 0
    missing: list[str] = []

    for flower in flowers:
        name = flower.get("englishName", "")
        row = _lookup(name, table)
        if row is None:
            missing.append(name)
            continue

        preservation = flower.setdefault("preservation", {})
        before = json.dumps(preservation, ensure_ascii=False, sort_keys=True)

        # --- ethylene sensitivity -------------------------------------------
        eth_text = (row.get("Ethylene") or {}).get("text", "").strip()
        if eth_text:
            preservation["ethyleneSensitivity"] = eth_text
            first_digit = next((c for c in eth_text if c.isdigit()), None)
            if first_digit:
                value = int(first_digit)
                preservation["ethyleneSensitivityScale"] = {
                    "value": eth_text,
                    "description": SENSITIVITY_DESCRIPTIONS.get(value, "unknown"),
                    "scale": SENSITIVITY_SCALE,
                }

        # --- farmer treatment (grower pre-treatment + Sugar) ----------------
        treatments = [
            _treatment(product, row[product])
            for product in FARMER_COLUMNS
            if (row.get(product) or {}).get("text", "").strip()
        ]
        preservation["farmerTreatment"] = treatments

        # --- ethylene blocker (STS) and leaf treatment ----------------------
        blocker_cell = row.get(ETHYLENE_BLOCKER_COLUMN) or {}
        preservation["ethyleneBlocker"] = (
            _treatment(ETHYLENE_BLOCKER_COLUMN, blocker_cell)
            if blocker_cell.get("text", "").strip()
            else None
        )

        leaf_cell = row.get(LEAF_TREATMENT_COLUMN) or {}
        preservation["leafTreatment"] = (
            _treatment(LEAF_TREATMENT_COLUMN, leaf_cell)
            if leaf_cell.get("text", "").strip()
            else None
        )

        # --- additives -------------------------------------------------------
        additives = (row.get("Additives") or {}).get("text", "").strip()
        preservation["additives"] = additives or None

        if json.dumps(preservation, ensure_ascii=False, sort_keys=True) != before:
            changed += 1
            if verbose:
                summary = ", ".join(
                    f"{t['productName']}={t['concentrationRate']}"
                    f"{'(rec)' if t['priority'] == 'recommended' else ''}"
                    for t in treatments
                )
                print(f"  {name}: {summary}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flowers, f, ensure_ascii=False, indent=2)

    return changed, missing


if __name__ == "__main__":
    count, not_in_doc = sync()
    print(f"\nUpdated {count} flowers from the DOC table.")
    if not_in_doc:
        print(f"No DOC row (left unchanged): {', '.join(not_in_doc)}")
