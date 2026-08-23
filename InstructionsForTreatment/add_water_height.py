"""Migration script: Add waterBucketHeight field to preservation section in flowers.json.

This script is idempotent - safe to run multiple times.

Rules:
- Rose (englishName == "Rose") gets waterBucketHeight = 50
- All other flowers get waterBucketHeight = null (None)
"""

from __future__ import annotations

import json
from pathlib import Path


def migrate(json_path: Path | str | None = None) -> int:
    """Add waterBucketHeight to every flower's preservation section.

    Args:
        json_path: Path to flowers.json. Defaults to ../flowers.json relative to this script.

    Returns:
        Number of flowers updated.
    """
    if json_path is None:
        json_path = Path(__file__).resolve().parent.parent / "flowers.json"
    else:
        json_path = Path(json_path)

    with open(json_path, encoding="utf-8") as f:
        flowers = json.load(f)

    updated = 0
    for flower in flowers:
        preservation = flower.get("preservation")
        if preservation is None:
            # Some flowers might not have a preservation section - add one
            flower["preservation"] = {"waterBucketHeight": None}
            updated += 1
            continue

        english_name = flower.get("englishName", "")
        if english_name == "Rose":
            preservation["waterBucketHeight"] = 50
        else:
            # Only set if not already present (idempotent)
            if "waterBucketHeight" not in preservation:
                preservation["waterBucketHeight"] = None

        updated += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flowers, f, ensure_ascii=False, indent=2)

    return updated


if __name__ == "__main__":
    count = migrate()
    print(f"Updated {count} flowers with waterBucketHeight field.")
