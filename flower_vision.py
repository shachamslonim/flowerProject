import os
import json
from pathlib import Path

from google.cloud import vision


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


def extract_flower_names(labels: list[vision.EntityAnnotation]) -> list[str]:
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
    for label in labels:
        text = label.description.lower()
        if any(keyword in text for keyword in flower_keywords):
            possible_flowers.append(label.description)

    return possible_flowers


def match_known_species(labels: list[vision.EntityAnnotation], flowers_json_path: Path) -> list[str]:
    """Matches Vision labels against flowers.json's englishName values only - e.g. Vision's
    "Garden roses" matches flowers.json's "Rose" ("Rose" is a substring of the label).
    Unlike extract_flower_names()'s fixed 18-keyword list (built for human-readable CLI
    display), this returns names InstructionsForTreatment's exact-match lookup can
    actually use, ready to pass as flower_name.

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
    for label in labels:
        text = label.description.lower()
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
    "all_labels": [{"description","score"}, ...], "source": "google_vision"}.
    confidence is None because label_detection returns per-label scores, not one
    flower-specific score - all_labels carries those for callers that want them.
    species_name is the first keyword-matched label (label_detection doesn't rank
    "flowerness", just generic label confidence); all_matches carries every match,
    not just the first, so callers (including main()'s printing) never need to
    re-run the keyword match themselves."""
    project_root = project_root or Path(__file__).parent.resolve()
    _ensure_credentials(project_root)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    response = client.label_detection(image=vision.Image(content=content))
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    labels = response.label_annotations
    all_labels = [{"description": label.description, "score": label.score} for label in labels]
    flower_names = extract_flower_names(labels)
    known_species_matches = match_known_species(labels, project_root / "flowers.json")

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
        "source": "google_vision",
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
