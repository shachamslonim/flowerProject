"""Tests for the machine-translation helper."""

import sys
from unittest.mock import MagicMock

import pytest

from InstructionsForTreatment.translate import translate_text


@pytest.fixture
def fake_deep_translator(monkeypatch):
    """Install a fake ``deep_translator`` module with a controllable translator."""
    translator = MagicMock()
    translator.translate.side_effect = lambda s: f"[he]{s}"
    gt_cls = MagicMock(return_value=translator)
    module = MagicMock(GoogleTranslator=gt_cls)
    monkeypatch.setitem(sys.modules, "deep_translator", module)
    return gt_cls, translator


def test_translates_text(fake_deep_translator):
    gt_cls, _ = fake_deep_translator
    assert translate_text("Sterilize the tools", target="he", source="en") == "[he]Sterilize the tools"
    # Hebrew maps to the legacy Google code "iw"
    assert gt_cls.call_args.kwargs == {"source": "en", "target": "iw"}


def test_empty_text_passthrough(fake_deep_translator):
    assert translate_text("   ", target="he") == "   "


def test_missing_dependency_returns_original(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "deep_translator":
            raise ImportError("no deep_translator")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "deep_translator", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    text = "Water bucket height: 50 cm"
    assert translate_text(text, target="he") == text


def test_service_failure_returns_original(monkeypatch):
    translator = MagicMock()
    translator.translate.side_effect = RuntimeError("service down")
    module = MagicMock(GoogleTranslator=MagicMock(return_value=translator))
    monkeypatch.setitem(sys.modules, "deep_translator", module)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    text = "Sterilize the tools.\n\nWater bucket height: 50 cm."
    assert translate_text(text, target="he", retries=2) == text


def test_paragraph_fallback_for_long_text(monkeypatch):
    translator = MagicMock()
    translator.translate.side_effect = lambda s: s.replace("cm", "ס\"מ")
    module = MagicMock(GoogleTranslator=MagicMock(return_value=translator))
    monkeypatch.setitem(sys.modules, "deep_translator", module)

    long_text = ("A" * 3000) + "\n\n" + "height 50 cm"
    out = translate_text(long_text, target="he")
    assert 'height 50 ס"מ' in out
