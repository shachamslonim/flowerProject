"""Tests for data_loader module."""

import pytest

from InstructionsForTreatment.data_loader import load_flower_data


class TestLoadFlowerData:
    """Tests for load_flower_data function."""

    def test_load_rose_by_exact_name(self):
        data = load_flower_data("Rose")
        assert data["id"] == 2
        assert data["englishName"] == "Rose"

    def test_load_rose_lowercase(self):
        data = load_flower_data("rose")
        assert data["id"] == 2

    def test_load_rose_uppercase(self):
        data = load_flower_data("ROSE")
        assert data["id"] == 2

    def test_load_anemone(self):
        data = load_flower_data("Anemone")
        assert data["id"] == 1
        assert data["englishName"] == "Anemone"

    def test_load_sunflower(self):
        data = load_flower_data("Sunflower")
        assert data["id"] == 3

    def test_flower_not_found_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            load_flower_data("NonExistentFlower")

    def test_flower_has_preservation_section(self):
        data = load_flower_data("Rose")
        assert "preservation" in data
        assert "farmerTreatment" in data["preservation"]
        assert "ethyleneSensitivity" in data["preservation"]

    def test_flower_has_hebrew_name(self):
        data = load_flower_data("Rose")
        assert data["hebrewName"] == "ורד"

    def test_whitespace_in_name_is_stripped(self):
        data = load_flower_data("  Rose  ")
        assert data["id"] == 2

    def test_file_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_flower_data("Rose", json_path="nonexistent.json")

    def test_rose_has_water_bucket_height(self):
        data = load_flower_data("Rose")
        assert data["preservation"]["waterBucketHeight"] == 50

    def test_anemone_water_bucket_height_is_null(self):
        data = load_flower_data("Anemone")
        assert data["preservation"]["waterBucketHeight"] is None

    def test_all_flowers_have_water_bucket_height_key(self):
        """Verify every flower has the waterBucketHeight field after migration."""
        import json
        from InstructionsForTreatment.data_loader import DEFAULT_FLOWERS_JSON

        with open(DEFAULT_FLOWERS_JSON, encoding="utf-8") as f:
            flowers = json.load(f)
        for flower in flowers:
            assert "waterBucketHeight" in flower.get("preservation", {}), (
                f"Missing waterBucketHeight for {flower.get('englishName', '?')}"
            )
