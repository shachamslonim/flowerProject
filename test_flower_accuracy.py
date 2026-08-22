import os
import json
from pathlib import Path

from google.cloud import vision


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def setup_credentials(project_root: Path) -> None:
    """Validate and set up Google Vision API credentials."""
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is missing in .env")

    credentials_path = Path(credentials)
    if not credentials_path.is_absolute():
        credentials_path = (project_root / credentials_path).resolve()

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}. "
            "Update GOOGLE_APPLICATION_CREDENTIALS in .env"
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)


def check_match(expected_name: str, texts: list[str]) -> bool:
    """Check if expected flower name appears (case-insensitive contains) in any text."""
    expected_lower = expected_name.lower()
    for text in texts:
        if expected_lower in text.lower():
            return True
    return False


def try_web_detection(client: vision.ImageAnnotatorClient, content: bytes) -> list[str]:
    """Try web detection to get more specific entity names."""
    image = vision.Image(content=content)
    response = client.web_detection(image=image)

    if response.error.message:
        return []

    web = response.web_detection
    entities = []
    if web.web_entities:
        for entity in web.web_entities:
            if entity.description:
                entities.append(entity.description)

    return entities


def run_test() -> None:
    """Run the flower vision accuracy test with 2-tier detection and retries."""
    project_root = Path(__file__).parent.resolve()

    # Load environment and credentials
    load_env_file(project_root / ".env")
    setup_credentials(project_root)

    # Get retry count from env
    max_retries = int(os.environ.get("RETRY_NOT_MATCHED", "0"))

    # Load test configuration
    test_file = project_root / "test_flowers.json"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_entries = json.loads(test_file.read_text(encoding="utf-8"))
    total = len(test_entries)

    print("=" * 60)
    print("  Flower Vision API Accuracy Test")
    print("=" * 60)
    print(f"\n  Tier 1: Label Detection (max 15 labels)")
    print(f"  Tier 2: Web Detection (fallback)")
    print(f"  Retries for not matched: {max_retries}")
    print(f"\n  Testing {total} flowers...\n")

    # Initialize Vision client
    client = vision.ImageAnnotatorClient()

    matched = 0
    not_matched = 0
    errors = 0
    match_tier = {"label": 0, "web": 0}
    not_matched_flowers = []
    # Track entries that failed for retry
    failed_entries = []

    for idx, entry in enumerate(test_entries, start=1):
        expected_name = entry["expected_name"]
        image_paths = entry.get("image_paths", [entry.get("image_path", "")])
        folder_name = entry["folder_name"]

        print(f"[{idx}/{total}] {folder_name} | Expected: {expected_name}")

        result = _test_flower(client, project_root, expected_name, image_paths)

        if result["status"] == "detected":
            print(f"  Result: \u2713 DETECTED (via {result['method']})\n")
            matched += 1
            match_tier[result["tier"]] += 1
        elif result["status"] == "error":
            print(f"  Result: SKIPPED (all photos had errors)\n")
            errors += 1
        else:
            print(f"  Result: \u2717 NOT DETECTED\n")
            not_matched += 1
            not_matched_flowers.append(f"{folder_name} ({expected_name})")
            failed_entries.append(entry)

    # === RETRIES ===
    for retry_round in range(1, max_retries + 1):
        if not failed_entries:
            break

        print("-" * 60)
        print(f"  RETRY ROUND {retry_round}/{max_retries} "
              f"({len(failed_entries)} flowers to retry)")
        print("-" * 60 + "\n")

        still_failed = []
        for entry in failed_entries:
            expected_name = entry["expected_name"]
            folder_name = entry["folder_name"]

            # Get additional photos from the folder beyond the ones already tested
            photos_already_tested = entry.get("image_paths", [entry.get("image_path", "")])
            num_tested = len(photos_already_tested)

            # Find more photos in the open folder
            open_folder = project_root / "photo" / folder_name / "open"
            if open_folder.exists():
                all_photos = sorted([
                    f.name for f in open_folder.iterdir()
                    if f.suffix.lower() == ".jpg" and f.name.startswith("open_")
                ])
                # Pick next batch of photos (skip already tested ones)
                start_idx = num_tested + (retry_round - 1) * 3
                retry_photos = [
                    f"photo/{folder_name}/open/{name}"
                    for name in all_photos[start_idx:start_idx + 3]
                ]
            else:
                retry_photos = []

            if not retry_photos:
                print(f"  [{folder_name}] No more photos available to retry")
                still_failed.append(entry)
                continue

            print(f"  [{folder_name}] | Expected: {expected_name} "
                  f"(trying {len(retry_photos)} new photos)")

            result = _test_flower(client, project_root, expected_name, retry_photos)

            if result["status"] == "detected":
                print(f"  Result: \u2713 DETECTED (via {result['method']})\n")
                matched += 1
                not_matched -= 1
                match_tier[result["tier"]] += 1
                not_matched_flowers.remove(f"{folder_name} ({expected_name})")
            else:
                print(f"  Result: \u2717 STILL NOT DETECTED\n")
                still_failed.append(entry)

        failed_entries = still_failed

    # Print summary
    successfully_tested = matched + not_matched
    accuracy = (matched / successfully_tested * 100) if successfully_tested > 0 else 0.0

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total tested:  {successfully_tested}")
    print(f"  Matched:       {matched}")
    print(f"    - via Label Detection:  {match_tier['label']}")
    print(f"    - via Web Detection:    {match_tier['web']}")
    print(f"  Not matched:   {not_matched}")
    if not_matched_flowers:
        for flower in not_matched_flowers:
            print(f"    - {flower}")
    print(f"  Errors:        {errors}")
    print(f"  Accuracy:      {accuracy:.1f}%")
    print("=" * 60)


def _test_flower(
    client: vision.ImageAnnotatorClient,
    project_root: Path,
    expected_name: str,
    image_paths: list[str],
) -> dict:
    """Test a single flower against multiple photos. Returns detection result."""
    all_photos_error = True

    for photo_idx, image_rel_path in enumerate(image_paths, start=1):
        image_path = (project_root / image_rel_path).resolve()
        photo_label = f"  Photo {photo_idx}/{len(image_paths)}"

        if not image_path.exists():
            print(f"{photo_label}: ERROR - Image not found: {image_path}")
            continue

        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            all_photos_error = False

            # === TIER 1: Label Detection ===
            image = vision.Image(content=content)
            response = client.label_detection(image=image, max_results=15)

            if response.error.message:
                print(f"{photo_label}: ERROR - {response.error.message}")
                continue

            labels = response.label_annotations
            label_names = [f"{l.description} ({l.score:.2f})" for l in labels]
            label_texts = [l.description for l in labels]
            print(f"{photo_label} [Tier 1] Labels: {', '.join(label_names)}")

            if check_match(expected_name, label_texts):
                matched_label = ""
                for l in labels:
                    if expected_name.lower() in l.description.lower():
                        matched_label = f"{l.description} ({l.score:.2f})"
                        break
                return {
                    "status": "detected",
                    "method": f"Label Detection - {matched_label}",
                    "tier": "label",
                }

            # === TIER 2: Web Detection ===
            print(f"{photo_label} [Tier 1] No match, trying Web Detection...")
            web_entities = try_web_detection(client, content)

            if web_entities:
                web_display = [e for e in web_entities[:10]]
                print(f"{photo_label} [Tier 2] Web entities: {', '.join(web_display)}")

                if check_match(expected_name, web_entities):
                    return {
                        "status": "detected",
                        "method": "Web Detection",
                        "tier": "web",
                    }

        except Exception as exc:
            print(f"{photo_label}: ERROR - {exc}")
            continue

    if all_photos_error:
        return {"status": "error", "method": "", "tier": ""}
    return {"status": "not_detected", "method": "", "tier": ""}


if __name__ == "__main__":
    run_test()
