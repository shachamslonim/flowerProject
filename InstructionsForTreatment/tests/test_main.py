"""Tests for main module and CLI integration."""

import pytest

from InstructionsForTreatment.main import generate_treatment_instructions


class TestGenerateTreatmentInstructions:
    """Integration tests for the main generate function."""

    def test_rose_english_open(self):
        result = generate_treatment_instructions(
            flower_name="Rose",
            is_open=True,
            yellow_leaves=False,
            leaves_falling=False,
            language="en",
            mode="template",
        )
        assert "Rose" in result
        assert "15 minutes" in result
        assert "50 cm" in result
        assert "acetone" in result

    def test_rose_hebrew_open(self):
        result = generate_treatment_instructions(
            flower_name="Rose",
            is_open=True,
            language="he",
        )
        assert "ורד" in result
        assert "15 דקות" in result
        assert "אצטון" in result

    def test_anemone_english_with_sts(self):
        result = generate_treatment_instructions(
            flower_name="Anemone",
            is_open=False,
            language="en",
        )
        assert "STS" in result
        assert "TOG-L-101" in result

    def test_invalid_flower_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            generate_treatment_instructions(
                flower_name="FakeFlower",
                is_open=False,
            )

    def test_invalid_language_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid language"):
            generate_treatment_instructions(
                flower_name="Rose",
                is_open=False,
                language="fr",
            )

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            generate_treatment_instructions(
                flower_name="Rose",
                is_open=False,
                mode="invalid",
            )

    def test_invalid_audience_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid audience"):
            generate_treatment_instructions(
                flower_name="Rose",
                is_open=False,
                audience="scientist",
            )

    def test_valid_audience_with_template_mode_is_ignored(self):
        """A valid audience is accepted even in template mode (no visible effect)."""
        result = generate_treatment_instructions(
            flower_name="Rose",
            is_open=False,
            language="en",
            mode="template",
            audience="agronomist",
        )
        assert "Rose" in result

    def test_llm_mode_falls_back_to_template(self):
        """LLM mode falls back to the template when Ollama is unavailable."""
        from unittest.mock import patch

        with patch("langchain_ollama.ChatOllama", side_effect=Exception("no ollama")):
            result = generate_treatment_instructions(
                flower_name="Rose",
                is_open=False,
                mode="llm",
            )
        # Fallback is the English template verbatim
        assert "Rose" in result
        assert "acetone" in result
        assert "50 cm" in result

    def test_case_insensitive_flower_name(self):
        result = generate_treatment_instructions(
            flower_name="rose",
            is_open=False,
            language="en",
        )
        assert "Rose" in result

    def test_default_parameters(self):
        """Test with minimal arguments using defaults."""
        result = generate_treatment_instructions(
            flower_name="Sunflower",
            is_open=False,
        )
        assert "Sunflower" in result
        # is_open=False renders the OPEN-flower wording (flag meaning is inverted) - open
        # flowers have no fixed water-entry window, only closed (controlled harvest) does
        assert "OPEN" in result
        assert "No fixed water entry time" in result
        assert "15 minutes" not in result

    def test_sunflower_no_sts(self):
        """A flower the DOC gives no TOG-L-101 rate for, at low sensitivity, gets no STS."""
        result = generate_treatment_instructions(
            flower_name="Achillea",
            is_open=False,
            language="en",
        )
        assert "TOG-L-101" not in result

    def test_doc_listed_blocker_is_shown_with_its_rate(self):
        """Sunflower is sensitivity 2 but the DOC lists TOG-L-101 0.5, so it shows."""
        result = generate_treatment_instructions(
            flower_name="Sunflower",
            is_open=False,
            language="en",
        )
        assert "ETHYLENE BLOCKER" in result
        assert "TOG-L-101" in result
        assert "concentration 0.5" in result


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_cli_parse_basic(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--flower", "Rose", "--open", "true", "--language", "en"])
        assert args.flower == "Rose"
        assert args.open is True
        assert args.language == "en"
        assert args.mode == "template"

    def test_cli_parse_defaults(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--flower", "Anemone"])
        assert args.flower == "Anemone"
        assert args.open is False
        assert args.yellow_leaves is False
        assert args.leaves_falling is False
        assert args.language == "en"
        assert args.mode == "template"

    def test_cli_parse_hebrew_llm(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--flower", "Lily", "--language", "he", "--mode", "llm"])
        assert args.language == "he"
        assert args.mode == "llm"

    def test_cli_parse_audience_default(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--flower", "Rose"])
        assert args.audience == "farmer"
        assert args.read is False
        assert args.speak_engine == "pyttsx3"
        assert args.llm_model is None
        assert args.translate is True

    def test_cli_parse_llm_model(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(
            ["--flower", "Rose", "--mode", "llm", "--llm-model", "aya-expanse:8b"]
        )
        assert args.llm_model == "aya-expanse:8b"

    def test_cli_parse_no_translate(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--flower", "Rose", "--no-translate"])
        assert args.translate is False

    def test_cli_parse_audience_and_read(self):
        from InstructionsForTreatment.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(
            ["--flower", "Rose", "--audience", "agronomist", "--read",
             "--speak-engine", "gtts"]
        )
        assert args.audience == "agronomist"
        assert args.read is True
        assert args.speak_engine == "gtts"
