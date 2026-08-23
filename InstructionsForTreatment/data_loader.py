"""Data loader for reading flower preservation data from flowers.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default path to flowers.json (parent directory of InstructionsForTreatment)
DEFAULT_FLOWERS_JSON = Path(__file__).resolve().parent.parent / "flowers.json"


def load_flower_data(flower_name: str, json_path: Path | str | None = None) -> dict[str, Any]:
    """Load flower data from flowers.json by English name (case-insensitive).

    Args:
        flower_name: The English name of the flower to look up.
        json_path: Optional path to the flowers.json file. Defaults to
                   the flowers.json in the parent project directory.

    Returns:
        The flower entry dict from the JSON array.

    Raises:
        ValueError: If the flower name is not found in the JSON.
        FileNotFoundError: If flowers.json does not exist at the given path.
    """
    path = Path(json_path) if json_path else DEFAULT_FLOWERS_JSON

    if not path.exists():
        raise FileNotFoundError(f"flowers.json not found at: {path}")

    with open(path, encoding="utf-8") as f:
        flowers = json.load(f)

    name_lower = flower_name.strip().lower()

    for flower in flowers:
        english_name = flower.get("englishName", "")
        if english_name.lower() == name_lower:
            return flower

    # Build list of available names for the error message
    available = [f.get("englishName", "?") for f in flowers[:10]]
    raise ValueError(
        f"Flower '{flower_name}' not found in {path.name}. "
        f"Available flowers include: {', '.join(available)}..."
    )
