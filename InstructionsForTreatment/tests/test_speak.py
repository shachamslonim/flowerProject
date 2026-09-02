"""Tests for the pluggable read-aloud package (InstructionsForTreatment.speak)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from InstructionsForTreatment.speak import (
    SPEAK_ENGINES,
    engine_extension,
    speak_text,
)
from InstructionsForTreatment.speak import (
    core,
    gtts_engine,
    mms_engine,
    pyttsx3_engine,
)
from InstructionsForTreatment.speak.player import play_file


def _fake_voice(voice_id, name=""):
    v = MagicMock()
    v.id = voice_id
    v.name = name
    return v


# ─── core ────────────────────────────────────────────────────────────────────


class TestCore:
    def test_engines_listed(self):
        assert set(SPEAK_ENGINES) == {"pyttsx3", "gtts", "mms"}

    def test_engine_extension(self):
        assert engine_extension("pyttsx3") == ".wav"
        assert engine_extension("gtts") == ".mp3"
        assert engine_extension("mms") == ".wav"

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid speak engine"):
            speak_text("hi", engine="espeak")

    def test_dispatches_to_engine_and_plays(self, tmp_path):
        out = tmp_path / "x.wav"
        with patch.object(pyttsx3_engine, "synthesize", return_value=out) as mock_syn, \
             patch.object(core, "play_file") as mock_play:
            result = speak_text("hello", out_path=out, engine="pyttsx3", play=True)
        assert result == out
        mock_syn.assert_called_once()
        mock_play.assert_called_once_with(out)

    def test_default_path_uses_engine_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        captured = {}

        def fake_syn(text, out_path, language, rate):
            captured["path"] = out_path
            return out_path

        with patch.object(gtts_engine, "synthesize", side_effect=fake_syn), \
             patch.object(core, "play_file"):
            speak_text("hello", engine="gtts", play=False)
        assert captured["path"].name == "instructions.mp3"

    def test_long_text_is_truncated(self, tmp_path):
        from InstructionsForTreatment.speak.core import MAX_SPEAK_CHARS

        captured = {}

        def fake_syn(text, out_path, language, rate):
            captured["len"] = len(text)
            return out_path

        with patch.object(pyttsx3_engine, "synthesize", side_effect=fake_syn), \
             patch.object(core, "play_file"):
            speak_text("x" * (MAX_SPEAK_CHARS + 500), out_path=tmp_path / "a.wav",
                       play=False)
        assert captured["len"] == MAX_SPEAK_CHARS

    def test_engine_unavailable_returns_none(self, tmp_path):
        with patch.object(pyttsx3_engine, "synthesize", return_value=None), \
             patch.object(core, "play_file") as mock_play:
            assert speak_text("hi", out_path=tmp_path / "a.wav") is None
        mock_play.assert_not_called()

    def test_prints_the_text_being_read(self, tmp_path, capsys):
        with patch.object(pyttsx3_engine, "synthesize", return_value=tmp_path / "a.wav"), \
             patch.object(core, "play_file"):
            speak_text("Sterilize the tools", out_path=tmp_path / "a.wav", play=False)
        err = capsys.readouterr().err
        assert "Reading aloud" in err
        assert "Sterilize the tools" in err


# ─── pyttsx3 engine ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_pyttsx3(monkeypatch):
    engine = MagicMock()
    engine.getProperty.return_value = [_fake_voice("en-US-david", "David")]
    module = MagicMock()
    module.init.return_value = engine
    monkeypatch.setitem(sys.modules, "pyttsx3", module)
    return module, engine


class TestPyttsx3Engine:
    def test_writes_wav(self, fake_pyttsx3, tmp_path):
        _, engine = fake_pyttsx3
        out = tmp_path / "out.wav"
        result = pyttsx3_engine.synthesize("hello world", out, "en", None)
        assert result == out
        args = engine.save_to_file.call_args[0]
        assert args == ("hello world", str(out))
        engine.runAndWait.assert_called_once()

    def test_rate_override(self, fake_pyttsx3, tmp_path):
        _, engine = fake_pyttsx3
        pyttsx3_engine.synthesize("hi", tmp_path / "a.wav", "en", 150)
        engine.setProperty.assert_any_call("rate", 150)

    def test_hebrew_voice_selected_when_available(self, monkeypatch, tmp_path):
        engine = MagicMock()
        engine.getProperty.return_value = [
            _fake_voice("en-US-david", "David"),
            _fake_voice("TOKENS\\MSHebrew", "Microsoft Hebrew Asaf"),
        ]
        module = MagicMock()
        module.init.return_value = engine
        monkeypatch.setitem(sys.modules, "pyttsx3", module)

        pyttsx3_engine.synthesize("שלום", tmp_path / "he.wav", "he", None)
        engine.setProperty.assert_any_call("voice", "TOKENS\\MSHebrew")

    def test_missing_pyttsx3_returns_none(self, monkeypatch, tmp_path):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "pyttsx3":
                raise ImportError("no pyttsx3")
            return real_import(name, *a, **k)

        monkeypatch.delitem(sys.modules, "pyttsx3", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert pyttsx3_engine.synthesize("hi", tmp_path / "a.wav", "en", None) is None

    def test_engine_failure_returns_none(self, monkeypatch, tmp_path):
        module = MagicMock()
        module.init.side_effect = RuntimeError("no speech engine")
        monkeypatch.setitem(sys.modules, "pyttsx3", module)
        assert pyttsx3_engine.synthesize("hi", tmp_path / "a.wav", "en", None) is None


# ─── gtts engine ─────────────────────────────────────────────────────────────


class TestGttsEngine:
    def test_writes_mp3_and_maps_hebrew_lang(self, monkeypatch, tmp_path):
        gtts_cls = MagicMock()
        instance = MagicMock()
        gtts_cls.return_value = instance
        fake_mod = MagicMock(gTTS=gtts_cls)
        monkeypatch.setitem(sys.modules, "gtts", fake_mod)

        out = tmp_path / "he.mp3"
        result = gtts_engine.synthesize("שלום", out, "he", None)

        assert result == out
        assert gtts_cls.call_args.kwargs["lang"] == "iw"
        instance.save.assert_called_once_with(str(out))

    def test_english_lang_passthrough(self, monkeypatch, tmp_path):
        gtts_cls = MagicMock()
        monkeypatch.setitem(sys.modules, "gtts", MagicMock(gTTS=gtts_cls))
        gtts_engine.synthesize("hi", tmp_path / "a.mp3", "en", None)
        assert gtts_cls.call_args.kwargs["lang"] == "en"

    def test_missing_gtts_returns_none(self, monkeypatch, tmp_path):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "gtts":
                raise ImportError("no gtts")
            return real_import(name, *a, **k)

        monkeypatch.delitem(sys.modules, "gtts", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert gtts_engine.synthesize("hi", tmp_path / "a.mp3", "en", None) is None

    def test_network_failure_returns_none(self, monkeypatch, tmp_path):
        gtts_cls = MagicMock(side_effect=Exception("connection failed"))
        monkeypatch.setitem(sys.modules, "gtts", MagicMock(gTTS=gtts_cls))
        assert gtts_engine.synthesize("hi", tmp_path / "a.mp3", "en", None) is None


# ─── mms (VITS / MMS-TTS) engine ─────────────────────────────────────────────


class TestMmsEngine:
    @pytest.fixture(autouse=True)
    def _no_token_read(self, monkeypatch):
        monkeypatch.setattr(mms_engine, "_load_hf_token", lambda: None)

    def _fake_model_and_tokenizer(self, is_uroman=False):
        import torch

        model = MagicMock()
        model.config.sampling_rate = 16000
        model.return_value.waveform = torch.zeros(1, 1600)
        tokenizer = MagicMock()
        tokenizer.is_uroman = is_uroman
        tokenizer.return_value = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        return model, tokenizer

    def test_writes_wav(self, monkeypatch, tmp_path):
        model, tok = self._fake_model_and_tokenizer()
        monkeypatch.setattr(mms_engine, "_get_model", lambda mid: (model, tok))
        out = tmp_path / "en.wav"
        with patch("scipy.io.wavfile.write") as mock_write:
            result = mms_engine.synthesize("hello", out, "en", None)
        assert result == out
        assert mock_write.call_args[0][0] == str(out)
        assert mock_write.call_args[0][1] == 16000

    def test_selects_hebrew_model(self, monkeypatch, tmp_path):
        seen = {}

        def fake_get(model_id):
            seen["id"] = model_id
            return self._fake_model_and_tokenizer()

        monkeypatch.setattr(mms_engine, "_get_model", fake_get)
        with patch("scipy.io.wavfile.write"):
            mms_engine.synthesize("שלום", tmp_path / "he.wav", "he", None)
        assert seen["id"] == "facebook/mms-tts-heb"

    def test_unknown_language_returns_none(self, tmp_path):
        assert mms_engine.synthesize("hi", tmp_path / "x.wav", "fr", None) is None

    def test_uroman_model_returns_none(self, monkeypatch, tmp_path):
        model, tok = self._fake_model_and_tokenizer(is_uroman=True)
        monkeypatch.setattr(mms_engine, "_get_model", lambda mid: (model, tok))
        assert mms_engine.synthesize("hi", tmp_path / "x.wav", "en", None) is None

    def test_model_load_failure_returns_none(self, monkeypatch, tmp_path):
        def boom(model_id):
            raise OSError("offline, cannot download")

        monkeypatch.setattr(mms_engine, "_get_model", boom)
        assert mms_engine.synthesize("hi", tmp_path / "x.wav", "en", None) is None


# ─── player ──────────────────────────────────────────────────────────────────


class TestPlayer:
    def test_wav_uses_winsound(self, monkeypatch, tmp_path):
        winsound = MagicMock()
        monkeypatch.setitem(sys.modules, "winsound", winsound)
        p = tmp_path / "a.wav"
        play_file(p)
        winsound.PlaySound.assert_called_once()

    def test_mp3_uses_startfile(self, monkeypatch, tmp_path):
        with patch("os.startfile", create=True) as mock_start:
            play_file(tmp_path / "a.mp3")
        mock_start.assert_called_once()

    def test_failure_is_swallowed(self, monkeypatch, tmp_path):
        with patch("os.startfile", create=True, side_effect=OSError("boom")):
            play_file(tmp_path / "a.mp3")  # must not raise
