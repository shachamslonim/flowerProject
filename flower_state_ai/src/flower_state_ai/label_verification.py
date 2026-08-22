"""Stage 1.5: Ollama vision re-verification of open/closed labels for species 001-015.

Species 001-015 hold 1,593 of the repo-wide 1,606 label_conflicts.csv rows (manifest/
filename/folder disagreements) - that's where the label noise lives, which is why they're
the ones re-verified here. review/ is a staging bucket the rest of the pipeline never
scans; folding it in through this stage reclaims thousands of otherwise-unused images.

Two-step workflow:
  1. run_verification() - resumable batch driver, calls the local Ollama vision model on
     every open/closed/review image for the configured species and appends one row per
     image to LABEL_VERIFICATION_CSV as it goes.
  2. merge_verified_labels() - backs up labels_master.csv once, then folds the verdicts
     back in (overwrite existing rows' resolved_state, insert new rows for review/ images),
     so every downstream stage keeps reading labels_master.csv exactly as before.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import ollama
import pandas as pd

from flower_state_ai import config
from flower_state_ai.labeling import SPECIES_DIR_RE, filename_state

VERIFICATION_PROMPT = (
    "You are labeling a flower photo's bloom state. OPEN means the petals are spread or "
    "unfurled and the reproductive parts (stamens/pistil) are visible. CLOSED means the "
    "flower is still a furled bud that has not bloomed yet. If no flower is clearly "
    "visible, or it is too blurry/ambiguous to judge, answer 'unclear'. If more than one "
    "flower is visible, judge the largest/most prominent one in the frame. Respond with "
    "only a JSON object of the form {\"state\": \"open\"|\"closed\"|\"unclear\", "
    "\"confidence\": \"low\"|\"medium\"|\"high\"}."
)

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": ["open", "closed", "unclear"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["state"],
}

_KEYWORD_RE = re.compile(r"\b(open|closed)\b", re.IGNORECASE)


def _parse_verdict(raw_text: str) -> tuple[str, str]:
    """Returns (state, parse_method). Falls back to a keyword scan if the model didn't
    return valid JSON (small vision models occasionally ignore the format constraint)."""
    try:
        parsed = json.loads(raw_text)
        state = str(parsed.get("state", "")).lower()
        if state in ("open", "closed", "unclear"):
            return state, "json"
    except (json.JSONDecodeError, AttributeError):
        pass

    hits = {m.group(1).lower() for m in _KEYWORD_RE.finditer(raw_text)}
    if hits == {"open"}:
        return "open", "keyword_fallback"
    if hits == {"closed"}:
        return "closed", "keyword_fallback"
    return "unclear", "keyword_fallback"


def classify_one_image(
    image_abs_path: str,
    model: str = config.OLLAMA_VISION_MODEL,
    max_retries: int = config.VERIFICATION_MAX_RETRIES,
) -> tuple[str, str, int, str | None]:
    """Returns (state, raw_response_text, attempts, error). Never raises - a bad image or
    a transient Ollama error must not kill a multi-hour batch run; on exhaustion this
    returns ("unclear", "", attempts, error_message)."""
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.generate(
                model=model,
                prompt=VERIFICATION_PROMPT,
                images=[image_abs_path],
                format=VERIFICATION_SCHEMA,
                options={"temperature": 0},
            )
            state, _parse_method = _parse_verdict(resp.response)
            return state, resp.response, attempt, None
        except Exception as e:  # noqa: BLE001 - any failure here must be recorded, not raised
            last_error = str(e)
            time.sleep(1)
    return "unclear", "", max_retries, last_error


def _scan_review_images(species_dir: Path, species_id: int, species_name: str) -> list[dict]:
    review_dir = species_dir / "review"
    if not review_dir.is_dir():
        return []
    rows = []
    for f in sorted(review_dir.iterdir()):
        if not f.is_file() or f.name.lower() == "manifest.jsonl":
            continue
        if f.suffix.lower() not in config.VALID_EXTS:
            continue
        rows.append({
            "image_path": str(f.relative_to(species_dir.parent)),
            "species_id": species_id,
            "species_name": species_name,
            "origin_folder": "review",
            "prior_resolved_state": None,
            "prior_source_used": None,
            "was_conflict": False,
            "filename_hint": filename_state(f.name),
        })
    return rows


def build_verification_worklist(
    photo_root: Path = config.PHOTO_ROOT,
    labels_master_csv: Path = config.LABELS_MASTER_CSV,
    conflicts_csv: Path = config.LABEL_CONFLICTS_CSV,
    species_ids: list[int] = config.VERIFY_SPECIES_IDS,
) -> pd.DataFrame:
    """Every existing open/closed labels_master.csv row for species_ids (full
    re-verification, not just conflicts) plus every review/ image for those species,
    scanned fresh from disk since review/ is never in labels_master.csv."""
    labels_df = pd.read_csv(labels_master_csv)
    scoped = labels_df[labels_df["species_id"].isin(species_ids)].copy()

    conflicts_df = pd.read_csv(conflicts_csv)
    conflicted_paths = set(conflicts_df["image_path"])

    existing_rows = [{
        "image_path": row.image_path,
        "species_id": row.species_id,
        "species_name": row.species_name,
        "origin_folder": row.folder_state,
        "prior_resolved_state": row.resolved_state,
        "prior_source_used": row.source_used,
        "was_conflict": row.image_path in conflicted_paths,
        "filename_hint": row.filename_state,
    } for row in scoped.itertuples(index=False)]

    review_rows: list[dict] = []
    species_dirs = sorted(p for p in photo_root.iterdir() if p.is_dir())
    for species_dir in species_dirs:
        m = SPECIES_DIR_RE.match(species_dir.name)
        if not m:
            continue
        species_id = int(m.group(1))
        if species_id not in species_ids:
            continue
        review_rows.extend(_scan_review_images(species_dir, species_id, m.group(2)))

    return pd.DataFrame(existing_rows + review_rows)


def run_verification(
    out_csv: Path = config.LABEL_VERIFICATION_CSV,
    model: str = config.OLLAMA_VISION_MODEL,
    limit: int | None = None,
    species_ids: list[int] = config.VERIFY_SPECIES_IDS,
    retry_errors: bool = False,
) -> pd.DataFrame:
    """Resumable batch driver: skips any image_path already present in out_csv (unless
    retry_errors and that row logged an error), appending+flushing after every image so an
    interruption loses at most one in-flight image."""
    worklist = build_verification_worklist(species_ids=species_ids)

    done_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(
        columns=["image_path", "species_id", "species_name", "origin_folder",
                 "prior_resolved_state", "was_conflict", "filename_hint",
                 "ollama_state", "raw_response", "attempts", "error"]
    )
    done_paths = set(done_df.loc[done_df["error"].isna(), "image_path"]) if not retry_errors \
        else set(done_df.loc[done_df["error"].isna() | (done_df["error"] == ""), "image_path"])

    todo = worklist[~worklist["image_path"].isin(done_paths)]
    if limit is not None:
        todo = todo.head(limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    total = len(todo)
    print(f"[verify-labels] {total} images to classify (model={model}, already done={len(done_paths)})")

    t_start = time.time()
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        abs_path = str(config.PHOTO_ROOT / row.image_path)
        state, raw, attempts, error = classify_one_image(abs_path, model=model)

        result_row = pd.DataFrame([{
            "image_path": row.image_path,
            "species_id": row.species_id,
            "species_name": row.species_name,
            "origin_folder": row.origin_folder,
            "prior_resolved_state": row.prior_resolved_state,
            "was_conflict": row.was_conflict,
            "filename_hint": row.filename_hint,
            "ollama_state": state,
            "raw_response": raw,
            "attempts": attempts,
            "error": error,
        }])
        result_row.to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False

        if i % 25 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = elapsed / i
            eta_min = rate * (total - i) / 60
            print(f"[verify-labels] {i}/{total} ({rate:.2f}s/img, ETA {eta_min:.1f} min)")

    return pd.read_csv(out_csv)


def merge_verified_labels(
    verification_csv: Path = config.LABEL_VERIFICATION_CSV,
    labels_master_csv: Path = config.LABELS_MASTER_CSV,
    backup_csv: Path = config.LABELS_MASTER_BACKUP_CSV,
    unclear_kept_csv: Path = config.VERIFICATION_REPORT_UNCLEAR_KEPT_CSV,
    review_unclear_csv: Path = config.VERIFICATION_REPORT_REVIEW_UNCLEAR_CSV,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Folds verified verdicts back into labels_master.csv. Existing open/closed rows get
    resolved_state overwritten unless the verdict is 'unclear' (never a valid
    CLASS_TO_IDX key - the prior label is kept and logged instead). review/ images with an
    open/closed verdict become new rows; review/ images verdicted 'unclear' are excluded
    and logged. Species outside the verified set pass through untouched."""
    verified_df = pd.read_csv(verification_csv)
    labels_df = pd.read_csv(labels_master_csv)

    existing_mask = verified_df["prior_resolved_state"].notna()
    existing = verified_df[existing_mask]
    review = verified_df[~existing_mask]

    existing_clear = existing[existing["ollama_state"] != "unclear"]
    existing_unclear = existing[existing["ollama_state"] == "unclear"]

    review_clear = review[review["ollama_state"].isin(["open", "closed"])]
    review_unclear = review[review["ollama_state"] == "unclear"]

    # review/ rows come from label_verification.csv, which can outlive the source photo
    # (e.g. the user later deletes an image that was verified in an earlier run) - filter
    # to files that still exist so a stale row can't inject a broken path into
    # labels_master.csv that preprocessing then fails to open.
    still_exists = review_clear["image_path"].map(lambda p: (config.PHOTO_ROOT / p).exists())
    stale_review = review_clear[~still_exists]
    review_clear = review_clear[still_exists]

    overrides = dict(zip(existing_clear["image_path"], existing_clear["ollama_state"]))
    changed_mask = labels_df["image_path"].isin(overrides) & (
        labels_df["image_path"].map(overrides).fillna(labels_df["resolved_state"]) != labels_df["resolved_state"]
    )
    n_changed = int(changed_mask.sum())

    new_labels_df = labels_df.copy()
    apply_mask = new_labels_df["image_path"].isin(overrides)
    new_labels_df.loc[apply_mask, "resolved_state"] = new_labels_df.loc[apply_mask, "image_path"].map(overrides)
    new_labels_df.loc[apply_mask, "source_used"] = "ollama_vision"

    new_review_rows = pd.DataFrame([{
        "image_path": r.image_path,
        "species_id": r.species_id,
        "species_name": r.species_name,
        "resolved_state": r.ollama_state,
        "source_used": "ollama_vision_review",
        "manifest_state": None,
        "filename_state": r.filename_hint,
        "folder_state": None,
        "manifest_sha1": None,
        "manifest_width": None,
        "manifest_height": None,
    } for r in review_clear.itertuples(index=False)])

    combined = pd.concat([new_labels_df, new_review_rows], ignore_index=True) if not new_review_rows.empty else new_labels_df

    print(f"[verify-labels] merge summary: changed={n_changed} added_from_review={len(new_review_rows)} "
          f"kept_prior_unclear={len(existing_unclear)} excluded_review_unclear={len(review_unclear)} "
          f"excluded_stale_review={len(stale_review)}")

    if dry_run:
        return combined

    if not backup_csv.exists():
        labels_df.to_csv(backup_csv, index=False)
        print(f"[verify-labels] backed up pre-verification labels -> {backup_csv}")

    unclear_kept_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_unclear.to_csv(unclear_kept_csv, index=False)
    review_unclear.to_csv(review_unclear_csv, index=False)

    combined.to_csv(labels_master_csv, index=False)
    print(f"[verify-labels] wrote {len(combined)} rows -> {labels_master_csv}")
    return combined


if __name__ == "__main__":
    run_verification()
    merge_verified_labels()
