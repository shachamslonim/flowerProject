import os
import json
import time
from pathlib import Path
from typing import Callable, TypeVar

from google.api_core import exceptions as vision_exceptions
from google.cloud import vision

T = TypeVar("T")

_TRANSIENT_VISION_ERRORS = (
    vision_exceptions.ServiceUnavailable,
    vision_exceptions.DeadlineExceeded,
    vision_exceptions.TooManyRequests,
    vision_exceptions.InternalServerError,
)


def _call_with_retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.0) -> T:
    """Retries fn() up to `attempts` times, only on a transient Vision API error
    (network blip, 429/503/500) - Vision is deterministic per image, so an in-band
    response.error (e.g. a malformed image) would just fail the same way on every
    attempt and is returned as-is for the caller to raise, not retried."""
    assert attempts >= 1
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except _TRANSIENT_VISION_ERRORS:
            if attempt == attempts:
                raise
            time.sleep(base_delay * attempt)
    raise AssertionError("unreachable")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_image_path(project_root: Path) -> Path:
    if len(os.sys.argv) > 1:
        return Path(os.sys.argv[1]).expanduser().resolve()

    env_image = os.environ.get("IMAGE_PATH", "")
    if env_image:
        image_path = Path(env_image)
        if not image_path.is_absolute():
            image_path = (project_root / image_path).resolve()
        return image_path

    return (project_root / "sample_flower.jpg").resolve()


def extract_flower_names(descriptions: list[str]) -> list[str]:
    flower_keywords = {
        "flower",
        "rose",
        "tulip",
        "sunflower",
        "lily",
        "orchid",
        "daisy",
        "lotus",
        "hibiscus",
        "peony",
        "lavender",
        "marigold",
        "jasmine",
        "dandelion",
        "chrysanthemum",
        "daffodil",
        "camellia",
        "poppy",
    }

    possible_flowers: list[str] = []
    for description in descriptions:
        text = description.lower()
        if any(keyword in text for keyword in flower_keywords):
            possible_flowers.append(description)

    return possible_flowers


def match_known_species(descriptions: list[str], flowers_json_path: Path) -> list[str]:
    """Matches Vision descriptions (label_detection labels or web_detection entities)
    against flowers.json's englishName values only - e.g. Vision's "Garden roses" matches
    flowers.json's "Rose" ("Rose" is a substring of the description). Unlike
    extract_flower_names()'s fixed 18-keyword list (built for human-readable CLI display),
    this returns names InstructionsForTreatment's exact-match lookup can actually use,
    ready to pass as flower_name.

    Deliberately does NOT match against commonNames or check the reverse direction (label
    found within the species name) - both were tried and produced real false positives:
    "Windflower" (Anemone's commonName) contains "flower", so the generic Vision label
    "Flower" matched Anemone; "Farewell-to-spring" (Clarkia's commonName) contains "spring",
    matching the season label "Spring"; and "Pink" (Dianthus's commonName, a real colloquial
    name for the flower) exactly collided with the color label "Pink". englishName-only,
    forward-only substring matching avoids all three without a fragile blocklist."""
    if not flowers_json_path.exists():
        return []
    with open(flowers_json_path, encoding="utf-8") as f:
        flowers = json.load(f)
    species_names = [flower["englishName"] for flower in flowers]

    matched: list[str] = []
    for description in descriptions:
        text = description.lower()
        for english_name in species_names:
            if english_name not in matched and english_name.lower() in text:
                matched.append(english_name)
    return matched


def _ensure_credentials(project_root: Path) -> Path:
    load_env_file(project_root / ".env")

    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is missing in .env")

    credentials_path = Path(credentials)
    if not credentials_path.is_absolute():
        credentials_path = (project_root / credentials_path).resolve()

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}. Update GOOGLE_APPLICATION_CREDENTIALS in .env"
        )

    try:
        raw_credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Credentials file is not valid JSON: {credentials_path}") from exc

    cred_type = raw_credentials.get("type", "<missing>")
    if cred_type != "service_account":
        raise RuntimeError(
            "Google Vision requires a Service Account key JSON. "
            f"Found credentials type: '{cred_type}' in file: {credentials_path.name}. "
            "If the filename starts with 'client_secret_', this is an OAuth client secret and cannot be used here. "
            "Create a Service Account key in Google Cloud Console and update GOOGLE_APPLICATION_CREDENTIALS in .env."
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    return credentials_path


def identify_flower(image_path: Path, project_root: Path | None = None) -> dict:
    """Programmatic entry point (the API/pipeline call this, not main()). Returns
    {"species_name": str | None, "confidence": None, "all_matches": [str, ...],
    "all_labels": [{"description","score"}, ...], "source": "google_vision_label" |
    "google_vision_web"}. confidence is None because label_detection returns per-label
    scores, not one flower-specific score - all_labels carries those for callers that
    want them. species_name is the first keyword-matched label (label_detection doesn't
    rank "flowerness", just generic label confidence); all_matches carries every match,
    not just the first, so callers (including main()'s printing) never need to re-run
    the keyword match themselves.

    Tier 1 (label_detection) runs first. Tier 2 (web_detection) only runs as a fallback
    when Tier 1's labels don't match any species in flowers.json - Vision is
    deterministic per image, so re-running the same call on the same bytes can't turn a
    miss into a hit; a different detection method can. Each tier retries up to 3 times,
    but only on a transient Vision API error (network blip, 429/503/500) - an in-band
    response.error (e.g. a malformed image) is the same on every attempt and isn't
    retried."""
    project_root = project_root or Path(__file__).parent.resolve()
    _ensure_credentials(project_root)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    flowers_json_path = project_root / "flowers.json"

    # Tier 1: Label Detection
    label_response = _call_with_retry(lambda: client.label_detection(image=image))
    if label_response.error.message:
        raise RuntimeError(f"Vision API error: {label_response.error.message}")

    labels = label_response.label_annotations
    label_descriptions = [label.description for label in labels]
    all_labels = [{"description": label.description, "score": label.score} for label in labels]
    flower_names = extract_flower_names(label_descriptions)
    known_species_matches = match_known_species(label_descriptions, flowers_json_path)
    source = "google_vision_label"

    # Tier 2: Web Detection, fallback only - Tier 1 found no flowers.json match
    if not known_species_matches:
        web_response = _call_with_retry(lambda: client.web_detection(image=image))
        if not web_response.error.message and web_response.web_detection.web_entities:
            web_descriptions = [
                entity.description for entity in web_response.web_detection.web_entities if entity.description
            ]
            web_species_matches = match_known_species(web_descriptions, flowers_json_path)
            if web_species_matches:
                known_species_matches = web_species_matches
                source = "google_vision_web"

    # "flower" is itself a keyword (so a generic "Flower" label counts as a match, which is
    # correct for the human-readable CLI listing), but it's useless as a species_name - a
    # downstream lookup (e.g. flowers.json) needs an actual, exact species name. Prefer a
    # flowers.json match (ready to use as-is); then any keyword match that isn't literally
    # "flower"; only fall back to "flower" itself if nothing more specific matched at all.
    specific_matches = [name for name in flower_names if name.strip().lower() != "flower"]
    species_name = (
        known_species_matches[0] if known_species_matches
        else specific_matches[0] if specific_matches
        else flower_names[0] if flower_names
        else None
    )

    return {
        "species_name": species_name,
        "confidence": None,
        "all_matches": flower_names,
        "known_species_matches": known_species_matches,
        "all_labels": all_labels,
        "source": source,
    }


def main() -> None:
    project_root = Path(__file__).parent.resolve()
    image_path = resolve_image_path(project_root)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Put a flower image in this path or pass one as an argument."
        )

    result = identify_flower(image_path, project_root)

    print("Detected Labels:")
    for label in result["all_labels"]:
        print(f"- {label['description']} (Confidence: {label['score']:.2f})")

    if result["all_matches"]:
        print("\nLikely Flower Name(s):")
        for name in result["all_matches"]:
            print(f"- {name}")
    else:
        print("\nNo specific flower name found. The image may not contain a clearly identifiable flower.")


if __name__ == "__main__":
    main()
