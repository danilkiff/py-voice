"""Tests for app.py."""

from __future__ import annotations

from pathlib import Path


import app
from app import (
    DEFAULT_MODEL,
    EMPTY_INPUT_MESSAGE,
    MODEL_CHOICES,
    build_app,
    format_header,
    make_run_handler,
    write_text_file,
)
from transcriber import TranscriptionResult


# ---------- constants ----------


class TestConstants:
    def test_model_choices_contains_expected_sizes(self):
        for size in ("tiny", "base", "small", "medium", "large-v3"):
            assert size in MODEL_CHOICES

    def test_default_model_is_in_choices(self):
        assert DEFAULT_MODEL in MODEL_CHOICES


# ---------- format_header ----------


class TestFormatHeader:
    def test_basic(self):
        result = TranscriptionResult(text="x", language="ru", duration=12.3)
        header = format_header(result, "large-v3")
        assert "Язык: ru" in header
        assert "Длительность: 12.3 с" in header
        assert "Модель: large-v3" in header
        assert header.endswith("\n\n")

    def test_zero_duration_rendered(self):
        result = TranscriptionResult(text="", language="ru", duration=0.0)
        header = format_header(result, "tiny")
        assert "0.0" in header

    def test_other_language_passes_through(self):
        result = TranscriptionResult(text="x", language="en", duration=1.0)
        assert "Язык: en" in format_header(result, "tiny")


# ---------- write_text_file ----------


class TestWriteTextFile:
    def test_writes_utf8_to_directory(self, tmp_path):
        out = write_text_file("audio", "привет мир", directory=tmp_path)
        assert out == tmp_path / "audio.txt"
        assert out.read_text(encoding="utf-8") == "привет мир"

    def test_overwrites_existing_file(self, tmp_path):
        write_text_file("a", "first", directory=tmp_path)
        out = write_text_file("a", "second", directory=tmp_path)
        assert out.read_text(encoding="utf-8") == "second"

    def test_default_directory_is_tempdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.tempfile.gettempdir", lambda: str(tmp_path))
        out = write_text_file("default", "x")
        assert out.parent == tmp_path
        assert out.name == "default.txt"


# ---------- make_run_handler ----------


def _fake_transcribe(audio_path: str, model_size: str) -> TranscriptionResult:
    return TranscriptionResult(
        text=f"text-of-{Path(audio_path).stem}", language="ru", duration=2.5
    )


class TestMakeRunHandler:
    def test_no_audio_returns_message_and_no_file(self, tmp_path):
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        text, file_path = run(None, "tiny")
        assert text == EMPTY_INPUT_MESSAGE
        assert file_path is None

    def test_empty_string_audio_returns_message(self, tmp_path):
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        text, file_path = run("", "tiny")
        assert text == EMPTY_INPUT_MESSAGE
        assert file_path is None

    def test_valid_input_returns_text_with_header(self, tmp_path):
        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"RIFF")
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        text, file_path = run(str(audio), "large-v3")
        assert "Язык: ru" in text
        assert "Модель: large-v3" in text
        assert "text-of-voice" in text
        assert file_path is not None
        assert Path(file_path).read_text(encoding="utf-8") == text

    def test_output_file_named_after_audio_stem(self, tmp_path):
        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"")
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        _, file_path = run(str(audio), "tiny")
        assert Path(file_path).name == "lecture.txt"

    def test_transcribe_fn_receives_args(self, tmp_path):
        seen: list[tuple[str, str]] = []

        def spy(audio_path: str, model_size: str) -> TranscriptionResult:
            seen.append((audio_path, model_size))
            return TranscriptionResult(text="ok", language="ru", duration=1.0)

        run = make_run_handler(transcribe_fn=spy, output_dir=tmp_path)
        run("/some/file.wav", "small")
        assert seen == [("/some/file.wav", "small")]

    def test_default_transcribe_fn_uses_get_transcriber(self, monkeypatch):
        called = {}

        def fake_default(audio_path, model_size):
            called["args"] = (audio_path, model_size)
            return TranscriptionResult(text="ok", language="ru", duration=1.0)

        monkeypatch.setattr(app, "_default_transcribe", fake_default)
        # Re-build the handler so it captures the patched default.
        run = app.make_run_handler()
        # We need a real path to satisfy the truthy check.
        run("/x.wav", "tiny")
        assert called["args"] == ("/x.wav", "tiny")


# ---------- _default_transcribe ----------


class TestDefaultTranscribe:
    def test_delegates_to_get_transcriber(self, monkeypatch):
        seen = {}

        class FakeTranscriber:
            def transcribe(self, audio_path):
                seen["audio"] = audio_path
                return TranscriptionResult(text="ok", language="ru", duration=1.0)

        def fake_get(model_size):
            seen["model_size"] = model_size
            return FakeTranscriber()

        monkeypatch.setattr(app, "get_transcriber", fake_get)
        result = app._default_transcribe("/x.mp3", "small")
        assert result.text == "ok"
        assert seen == {"audio": "/x.mp3", "model_size": "small"}


# ---------- build_app smoke test ----------


class TestBuildApp:
    def test_returns_gradio_blocks(self):
        import gradio as gr

        instance = build_app(transcribe_fn=_fake_transcribe)
        assert isinstance(instance, gr.Blocks)

    def test_default_call_does_not_raise(self):
        # No transcribe_fn argument — uses module default.
        import gradio as gr

        instance = build_app()
        assert isinstance(instance, gr.Blocks)
