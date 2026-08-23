"""Tests for LLM renderer module."""

import pytest
from unittest.mock import patch, MagicMock

from InstructionsForTreatment.data_loader import load_flower_data
from InstructionsForTreatment.llm_renderer import (
    render_instructions_llm,
    _build_data_prompt,
    OLLAMA_MODEL,
)


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

    def test_prompt_includes_environment(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "20°C" in prompt
        assert "30%" in prompt

    def test_prompt_includes_water_height(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "50 cm" in prompt

    def test_prompt_includes_water_entry_when_open(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")
        assert "15 minutes" in prompt
        assert "OPEN" in prompt

    def test_prompt_no_water_entry_when_closed(self):
        data = load_flower_data("Rose")
        prompt = _build_data_prompt(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")
        assert "CLOSED" in prompt

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


class TestRenderInstructionsLLM:
    """Tests for render_instructions_llm with mocked LLM."""

    @patch("langchain_ollama.ChatOllama")
    def test_successful_llm_call(self, mock_ollama_class):
        """Test that LLM is called correctly and response is returned."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Professional treatment instructions for Rose..."
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        result = render_instructions_llm(data, is_open=True, yellow_leaves=False, leaves_falling=False, language="en")

        assert result == "Professional treatment instructions for Rose..."
        mock_ollama_class.assert_called_once_with(model=OLLAMA_MODEL, temperature=0.3)
        mock_llm.invoke.assert_called_once()

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
        mock_response.content = "Instructions..."
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="en")

        # Check the messages passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "English" in system_msg.content

    @patch("langchain_ollama.ChatOllama")
    def test_language_he_uses_hebrew_system_prompt(self, mock_ollama_class):
        """Verify Hebrew system prompt is used for language='he'."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "הוראות..."
        mock_llm.invoke.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        data = load_flower_data("Rose")
        render_instructions_llm(data, is_open=False, yellow_leaves=False, leaves_falling=False, language="he")

        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "עברית" in system_msg.content
