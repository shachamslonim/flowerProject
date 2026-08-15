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


def main() -> None:
    project_root = Path(__file__).parent.resolve()
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

    image_path = resolve_image_path(project_root)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Put a flower image in this path or pass one as an argument."
        )

    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    labels = response.label_annotations

    print("Detected Labels:")
    for label in labels:
        print(f"- {label.description} (Confidence: {label.score:.2f})")

    flower_names = extract_flower_names(labels)
    if flower_names:
        print("\nLikely Flower Name(s):")
        for name in flower_names:
            print(f"- {name}")
    else:
        print("\nNo specific flower name found. The image may not contain a clearly identifiable flower.")


if __name__ == "__main__":
    main()
