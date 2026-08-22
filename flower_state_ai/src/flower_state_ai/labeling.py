"""Stage 1: build a trustworthy open/closed label for every training image under photo/.

Label resolution priority per file (most to least authoritative):
1. manifest.jsonl entry's "state" field, if this file has one.
2. "open_"/"closed_" filename prefix, if present.
3. The folder (closed/ or open/) the file is physically sitting in.

review/ is never scanned - it's an unsorted staging bucket. Any disagreement between
available signals is logged to label_conflicts.csv (resolution still proceeds per the
priority above; nothing is silently guessed).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flower_state_ai import config

SPECIES_DIR_RE = re.compile(r"^(\d{3})_(.+)$")
FILENAME_STATE_RE = re.compile(r"^(open|closed)_", re.IGNORECASE)


def parse_species_dir(name: str) -> tuple[int, str] | None:
    m = SPECIES_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def load_manifest(folder: Path) -> dict[str, dict]:
    manifest_path = folder / "manifest.jsonl"
    if not manifest_path.exists():
        return {}
    records: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = rec.get("file")
            if fname:
                records[fname] = rec
    return records


def filename_state(fname: str) -> str | None:
    m = FILENAME_STATE_RE.match(fname)
    if not m:
        return None
    return m.group(1).lower()


@dataclass
class LabelResolution:
    resolved_state: str
    source_used: str
    manifest_state: str | None
    filename_state: str | None
    folder_state: str


def resolve_label(fname: str, folder_state: str, manifest_entry: dict | None) -> LabelResolution:
    manifest_state = None
    if manifest_entry is not None:
        ms = manifest_entry.get("state")
        if ms:
            manifest_state = str(ms).lower()

    fn_state = filename_state(fname)

    if manifest_state is not None:
        resolved_state, source_used = manifest_state, "manifest"
    elif fn_state is not None:
        resolved_state, source_used = fn_state, "filename"
    else:
        resolved_state, source_used = folder_state, "folder"

    return LabelResolution(
        resolved_state=resolved_state,
        source_used=source_used,
        manifest_state=manifest_state,
        filename_state=fn_state,
        folder_state=folder_state,
    )


def _conflict_pairs(res: LabelResolution) -> list[str]:
    signals = {
        "manifest": res.manifest_state,
        "filename": res.filename_state,
        "folder": res.folder_state,
    }
    names = list(signals)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            va, vb = signals[a], signals[b]
            if va is not None and vb is not None and va != vb:
                pairs.append(f"{a}_vs_{b}")
    return pairs


def build_labels_master(
    photo_root: Path = config.PHOTO_ROOT,
    out_csv: Path = config.LABELS_MASTER_CSV,
    conflicts_csv: Path = config.LABEL_CONFLICTS_CSV,
    skipped_csv: Path = config.SKIPPED_FILES_CSV,
) -> pd.DataFrame:
    rows: list[dict] = []
    conflict_rows: list[dict] = []
    skipped_rows: list[dict] = []

    species_dirs = sorted(p for p in photo_root.iterdir() if p.is_dir())
    for species_dir in species_dirs:
        parsed = parse_species_dir(species_dir.name)
        if parsed is None:
            continue
        species_id, species_name = parsed

        for state_dir_name in sorted(config.STATE_DIRS):
            state_dir = species_dir / state_dir_name
            if not state_dir.is_dir():
                continue
            manifest = load_manifest(state_dir)

            for f in sorted(state_dir.iterdir()):
                if not f.is_file() or f.name.lower() == "manifest.jsonl":
                    continue
                ext = f.suffix.lower()
                if ext not in config.VALID_EXTS:
                    skipped_rows.append({
                        "image_path": str(f.relative_to(photo_root)),
                        "reason": f"unsupported_extension:{ext or 'none'}",
                    })
                    continue

                manifest_entry = manifest.get(f.name)
                res = resolve_label(f.name, state_dir_name, manifest_entry)
                rel_path = str(f.relative_to(photo_root))

                rows.append({
                    "image_path": rel_path,
                    "species_id": species_id,
                    "species_name": species_name,
                    "resolved_state": res.resolved_state,
                    "source_used": res.source_used,
                    "manifest_state": res.manifest_state,
                    "filename_state": res.filename_state,
                    "folder_state": res.folder_state,
                    "manifest_sha1": manifest_entry.get("sha1") if manifest_entry else None,
                    "manifest_width": manifest_entry.get("width") if manifest_entry else None,
                    "manifest_height": manifest_entry.get("height") if manifest_entry else None,
                })

                conflict_types = _conflict_pairs(res)
                if conflict_types:
                    conflict_rows.append({
                        "image_path": rel_path,
                        "species_id": species_id,
                        "species_name": species_name,
                        "resolved_state": res.resolved_state,
                        "source_used": res.source_used,
                        "manifest_state": res.manifest_state,
                        "filename_state": res.filename_state,
                        "folder_state": res.folder_state,
                        "conflict_type": ";".join(conflict_types),
                    })

    labels_df = pd.DataFrame(rows)
    conflicts_df = pd.DataFrame(conflict_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_csv, index=False)
    conflicts_df.to_csv(conflicts_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    n_species = labels_df["species_id"].nunique() if not labels_df.empty else 0
    print(f"[labeling] wrote {len(labels_df)} labeled rows across {n_species} species -> {out_csv}")
    print(f"[labeling] {len(conflicts_df)} label conflicts -> {conflicts_csv}")
    print(f"[labeling] {len(skipped_df)} skipped files -> {skipped_csv}")
    return labels_df


if __name__ == "__main__":
    build_labels_master()
