"""Tests for LLM renderer module."""

import pytest
from unittest.mock import patch, MagicMock

from InstructionsForTreatment.data_loader import load_flower_data
from InstructionsForTreatment.llm_renderer import (
    render_instructions_llm,
    resolve_model,
    _build_data_prompt,
    _build_system_prompt,
    _looks_complete,
    OLLAMA_MODEL,
)

# A response that passes the truncation guard (>= 350 chars, >= 5 newlines).
_OK = (
    "1. Sterilization: clean the bins, shears, tables and guillotine with acetone.\n"
    "2. Materials: choose one biocide option and mix it into the water solution.\n"
    "3. Leaf treatment: this flower needs none.\n"
    "4. Environmental conditions: hold the flowers at 20C with 30% humidity.\n"
    "5. Water bucket: fill it to the height given for this flower.\n"
    "6. Water entry: get the stems into water within 15 minutes of harvest.\n"
    "7. Keep everything clean and check the flowers regularly.\n"
)
_OK_HE = _OK


class TestResolveModel:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert resolve_model() == OLLAMA_MODEL

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "aya-expanse:8b")
        assert resolve_model() == "aya-expanse:8b"

    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "gemma2:9b")
        assert resolve_model("command-r") == "command-r"


class TestBuildDataPrompt:
    """Tests for the data prompt builder."""

    def test_prompt_includes_sterilization_tools(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "bins" in prompt
        assert "shears" in prompt
        assert "tables" in prompt
        assert "guillotine" in prompt
        assert "acetone" in prompt

    def test_prompt_includes_materials(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "TOG-6" in prompt
        assert "0.0089" in prompt

    def test_prompt_frames_biocides_as_one_choice_with_priority(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "list ALL" in prompt and "use only ONE" in prompt
        # Rose: TOG-6 is recommended, TOG-30 is standard
        assert "TOG-6 at 0.0089 (recommended)" in prompt
        assert "TOG-30 at 0.035 (standard)" in prompt

    def test_prompt_sugar_is_an_add_on_not_a_choice(self):
        data = load_flower_data("Gladiolus")  # has Sugar in farmerTreatment
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "Sugar at 5-10 can be added on top" in prompt
        assert "add-on, not one of the choices" in prompt

    def test_prompt_includes_environment(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "20°C" in prompt
        assert "30%" in prompt

    def test_prompt_includes_water_height(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "50 cm" in prompt

    def test_prompt_open_wording(self):
        # is_open=False renders the OPEN-flower wording (flag meaning is inverted)
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "15 minutes" in prompt
        assert "flower is open" in prompt.lower()

    def test_prompt_closed_wording_still_states_the_time(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "flower is closed" in prompt.lower()
        # the template always gives the 15-minute time; the data prompt must too
        assert "15 minutes" in prompt

    def test_prompt_includes_sts_for_high_sensitivity(self):
        data = load_flower_data("Anemone")  # sensitivity=3
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "STS" in prompt
        assert "TOG-L-101" in prompt

    def test_prompt_no_sts_for_low_sensitivity(self):
        data = load_flower_data("Rose")  # sensitivity=1
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "TOG-L-101" not in prompt

    def test_prompt_includes_gibberellin_when_yellow_leaves(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=True, leaves_falling=False, language="en")
        assert "Gibberellin" in prompt

    def test_prompt_hebrew_includes_hebrew_name(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="he")
        assert "ורד" in prompt

    def test_prompt_english_includes_english_name(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "Rose" in prompt


class TestBuildSystemPrompt:
    """Tests for audience-tailored system prompt construction."""

    def test_farmer_guidance_en(self):
        prompt = _build_system_prompt("en", "farmer")
        assert "imperative steps" in prompt
        assert "field" in prompt

    def test_agronomist_guidance_en(self):
        prompt = _build_system_prompt("en", "agronomist")
        assert "agronomist" in prompt
        assert "technical" in prompt

    def test_layperson_guidance_en(self):
        prompt = _build_system_prompt("en", "layperson")
        assert "no agricultural background" in prompt
        assert "everyday words" in prompt

    def test_guidance_he_is_hebrew(self):
        prompt = _build_system_prompt("he", "agronomist")
        assert "אגרונום" in prompt
        assert "כתוב בעברית" in prompt

    def test_unknown_audience_falls_back_to_farmer(self):
        assert _build_system_prompt("en", "bogus") == _build_system_prompt("en", "farmer")


class TestRenderInstructionsLLM:
    """Tests for render_instructions_llm with mocked LLM."""

    @patch("langchain_ollama.ChatOllama")
    def test_successful_llm_call(self, mock_ollama_class):
        """Test that LLM is called correctly and response is returned."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = _OK
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")

        assert result == _OK.strip()
        assert mock_ollama_class.call_args.kwargs["model"] == OLLAMA_MODEL
        assert mock_ollama_class.call_args.kwargs["temperature"] == 0.3
        assert mock_ollama_class.call_args.kwargs["repeat_penalty"] > 1.0
        mock_llm.invoke.assert_called_once()

    @patch("langchain_ollama.ChatOllama")
    def test_model_argument_overrides_default(self, mock_ollama_class):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=_OK)
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False,
            language="he", model="aya-expanse:8b", translate=False,
        )
        assert mock_ollama_class.call_args.kwargs["model"] == "aya-expanse:8b"

    @patch("langchain_ollama.ChatOllama")
    def test_model_env_var_used_when_no_arg(self, mock_ollama_class, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "gemma2:9b")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=_OK)
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en"
        )
        assert mock_ollama_class.call_args.kwargs["model"] == "gemma2:9b"

    @patch("InstructionsForTreatment.translate.translate_text")
    @patch("langchain_ollama.ChatOllama")
    def test_hebrew_composes_in_english_then_translates(
        self, mock_ollama_class, mock_translate
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=_OK)
        mock_ollama_class.return_value = mock_llm
        mock_translate.return_value = "הוראות בעברית"

        data = load_flower_data("Rose")
        result = render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False,
            language="he", audience="farmer",
        )

        # composed with the English system prompt
        system_msg = mock_llm.invoke.call_args[0][0][0]
        assert "Write in English" in system_msg.content
        # and the English output was translated to Hebrew
        mock_translate.assert_called_once_with(_OK.strip(), target="he", source="en")
        assert result == "הוראות בעברית"

    @patch("InstructionsForTreatment.translate.translate_text")
    @patch("langchain_ollama.ChatOllama")
    def test_no_translate_trusts_model_hebrew(self, mock_ollama_class, mock_translate):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=_OK)
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False,
            language="he", translate=False,
        )

        system_msg = mock_llm.invoke.call_args[0][0][0]
        assert "כתוב בעברית" in system_msg.content
        mock_translate.assert_not_called()
        assert result == _OK.strip()

    @patch("InstructionsForTreatment.translate.translate_text")
    @patch("langchain_ollama.ChatOllama")
    def test_english_is_not_translated(self, mock_ollama_class, mock_translate):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=_OK)
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en"
        )
        mock_translate.assert_not_called()

    @patch("langchain_ollama.ChatOllama")
    def test_fallback_on_connection_error(self, mock_ollama_class):
        """Test fallback to template when Ollama connection fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("Cannot connect to Ollama")
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")

        # Should fall back to template and still produce valid output
        assert "Rose" in result
        assert "acetone" in result
        assert "50 cm" in result

    def test_looks_complete(self):
        assert _looks_complete(_OK)
        assert not _looks_complete("**Sterilization**\nClean the tools.\nStop.")
        assert not _looks_complete("")

    @patch("langchain_ollama.ChatOllama")
    def test_retries_then_falls_back_on_truncated_output(self, mock_ollama_class):
        """A model that keeps emitting a stub -> retry once -> template fallback."""
        stub = MagicMock(content="**Sterilization**\nWipe tools.\nDone.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = stub
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(
            data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en"
        )

        assert mock_llm.invoke.call_count == 2  # tried twice
        assert "=== Treatment Instructions for Rose ===" in result  # template fallback
        assert "50 cm" in result

    @patch("langchain_ollama.ChatOllama")
    def test_second_attempt_succeeds(self, mock_ollama_class):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [MagicMock(content="stub\nstub"), MagicMock(content=_OK)]
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(
            data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en"
        )
        assert result == _OK.strip()
        assert mock_llm.invoke.call_count == 2

    @patch("langchain_ollama.ChatOllama")
    def test_fallback_on_generic_exception(self, mock_ollama_class):
        """Test fallback to template on any exception."""
        mock_ollama_class.side_effect = Exception("Something went wrong")

        data = load_flower_data("Rose")
        result = render_instructions_llm(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="he")

        # Should fall back to Hebrew template
        assert "ורד" in result
        assert "אצטון" in result

    @patch("langchain_ollama.ChatOllama")
    def test_language_en_uses_english_system_prompt(self, mock_ollama_class):
        """Verify English system prompt is used for language='en'."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = _OK
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")

        # Check the messages passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "English" in system_msg.content

    @patch("langchain_ollama.ChatOllama")
    def test_language_he_uses_hebrew_system_prompt_when_not_translating(
        self, mock_ollama_class
    ):
        """With translate=False the Hebrew system prompt is used for language='he'."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = _OK
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False,
            language="he", translate=False,
        )

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "עברית" in system_msg.content

    @patch("langchain_ollama.ChatOllama")
    def test_audience_guidance_reaches_system_prompt(self, mock_ollama_class):
        """The chosen audience's guidance is included in the system message."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = _OK
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(
            data, is_open=False, yellow_leaves=False, leaves_falling=False,
            language="en", audience="agronomist",
        )

        system_msg = mock_llm.invoke.call_args[0][0][0]
        assert "agronomist" in system_msg.content
