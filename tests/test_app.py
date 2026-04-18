"""Tests for app.py."""

from __future__ import annotations

from pathlib import Path

import app
from app import (
    DEFAULT_MODEL,
    EMPTY_INPUT_MESSAGE,
    EMPTY_TRANSCRIPT_MESSAGE,
    EMPTY_URL_MESSAGE,
    MODEL_CHOICES,
    OLLAMA_UNAVAILABLE,
    YOUTUBE_ERROR,
    _build_thumbnail_html,
    _format_timecode,
    _group_timed_segments,
    _streaming_reduce,
    build_app,
    format_header,
    make_run_handler,
    make_summarize_handler,
    make_youtube_handler,
    write_text_file,
)
from transcriber import TranscriptionResult


def _last(gen):
    """Consume a generator and return its last yielded value."""
    value = None
    for value in gen:
        pass
    return value


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
        text, file_path = _last(run(None, "tiny"))
        assert text == EMPTY_INPUT_MESSAGE
        assert file_path is None

    def test_empty_string_audio_returns_message(self, tmp_path):
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        text, file_path = _last(run("", "tiny"))
        assert text == EMPTY_INPUT_MESSAGE
        assert file_path is None

    def test_valid_input_returns_text_with_header(self, tmp_path):
        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"RIFF")
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        text, file_path = _last(run(str(audio), "large-v3"))
        assert "Язык: ru" in text
        assert "Модель: large-v3" in text
        assert "text-of-voice" in text
        assert file_path is not None
        assert Path(file_path).read_text(encoding="utf-8") == text

    def test_output_file_named_after_audio_stem(self, tmp_path):
        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"")
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        _, file_path = _last(run(str(audio), "tiny"))
        assert Path(file_path).name == "lecture.txt"

    def test_transcribe_fn_receives_args(self, tmp_path):
        seen: list[tuple[str, str]] = []

        def spy(audio_path: str, model_size: str) -> TranscriptionResult:
            seen.append((audio_path, model_size))
            return TranscriptionResult(text="ok", language="ru", duration=1.0)

        run = make_run_handler(transcribe_fn=spy, output_dir=tmp_path)
        _last(run("/some/file.wav", "small"))
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
        _last(run("/x.wav", "tiny"))
        assert called["args"] == ("/x.wav", "tiny")

    def test_yields_progress_before_result(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        run = make_run_handler(transcribe_fn=_fake_transcribe, output_dir=tmp_path)
        steps = list(run(str(audio), "tiny"))
        assert len(steps) == 2
        assert steps[0] == ("Транскрибация…", None)


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


# ---------- _default_summarize ----------


class TestDefaultSummarize:
    def test_delegates_to_summarizer(self, monkeypatch):
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))
        calls = []

        class FakeSummarizer:
            def __init__(self, base_url, model):
                self.base_url = base_url
                self.model = model

            def summarize(self, text):
                calls.append((self.base_url, self.model, text))
                return "result"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)
        result = app._default_summarize("input")
        assert result == "result"
        assert calls == [("http://h:1", "m", "input")]


# ---------- _format_timecode ----------


class TestFormatTimecode:
    def test_zero(self):
        assert _format_timecode(0.0) == "0:00"

    def test_seconds_only(self):
        assert _format_timecode(5.0) == "0:05"

    def test_minutes_and_seconds(self):
        assert _format_timecode(65.0) == "1:05"

    def test_hours(self):
        assert _format_timecode(3661.0) == "1:01:01"

    def test_fractional_truncated(self):
        assert _format_timecode(90.9) == "1:30"


# ---------- _build_thumbnail_html ----------


class TestBuildThumbnailHtml:
    def test_returns_img_tag(self):
        from youtube import StoryboardInfo

        sb = StoryboardInfo(
            fragments=(("https://cdn/M0.jpg", 200.0),),
            width=160,
            height=90,
            rows=10,
            columns=10,
            fps=0.5,
        )
        html = _build_thumbnail_html(sb, 0.0)
        assert "<img" in html
        assert "https://cdn/M0.jpg" in html
        assert "object-fit:none" in html

    def test_returns_empty_for_none(self):
        assert _build_thumbnail_html(None, 0.0) == ""


# ---------- _group_timed_segments ----------


class TestGroupTimedSegments:
    def test_single_group(self):
        segs = ((0.0, "short"), (1.0, "text"))
        groups = _group_timed_segments(segs, chunk_size=100)
        assert len(groups) == 1
        assert groups[0][0] == 0.0
        assert "short" in groups[0][1]
        assert "text" in groups[0][1]

    def test_multiple_groups(self):
        segs = ((0.0, "a" * 50), (10.0, "b" * 50), (20.0, "c" * 50))
        groups = _group_timed_segments(segs, chunk_size=60)
        assert len(groups) == 3
        assert groups[0][0] == 0.0
        assert groups[1][0] == 10.0
        assert groups[2][0] == 20.0

    def test_empty_segments(self):
        assert _group_timed_segments((), chunk_size=100) == []

    def test_timestamps_from_first_segment_in_group(self):
        segs = ((5.0, "a" * 10), (15.0, "b" * 10), (25.0, "c" * 10))
        groups = _group_timed_segments(segs, chunk_size=30)
        assert groups[0][0] == 5.0


# ---------- _streaming_reduce ----------


class TestStreamingReduce:
    def test_short_combined_single_call(self):
        """Combined text fits in MAX_REDUCE_INPUT → one progress + one result."""
        results = list(_streaming_reduce("short", lambda t: "done", "pre\n"))
        assert results[-1] == "done"
        assert any("Финальная суммаризация" in r for r in results)

    def test_long_combined_shows_progress(self, monkeypatch):
        """Combined text exceeds MAX_REDUCE_INPUT → intermediate progress."""
        monkeypatch.setattr("map_reduce.MAX_REDUCE_INPUT", 50)
        monkeypatch.setattr("map_reduce.MAX_REDUCE_DEPTH", 2)

        calls: list[str] = []

        def shrink(text: str) -> str:
            calls.append(text)
            return "s"

        big = "x" * 200
        results = list(_streaming_reduce(big, shrink, ""))
        assert results[-1] == "s"
        # Should show intermediate "(N/M)" progress
        assert any("Формирование итога…" in r and "/" in r for r in results)
        assert len(calls) > 1

    def test_prefix_preserved(self):
        results = list(_streaming_reduce("text", lambda t: "r", "PREFIX\n"))
        assert any(r.startswith("PREFIX\n") for r in results[:-1])


# ---------- _default_youtube_summarize ----------


class TestDefaultYoutubeSummarize:
    def test_short_text_single_summary(self, monkeypatch):
        """Short subtitle text (< threshold) → single summarize call."""
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))

        class FakeSummarizer:
            def __init__(self, *a):
                pass

            def summarize(self, text):
                return f"sum({text[:10]})"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)

        from youtube import SubtitleResult, VideoInfo

        monkeypatch.setattr(
            "youtube.fetch_video_info",
            lambda url, **kw: VideoInfo(
                subtitles=SubtitleResult(text="short", language="ru"),
                storyboard=None,
            ),
        )
        monkeypatch.setattr("youtube.validate_youtube_url", lambda u: u)

        results = list(app._default_youtube_summarize("https://youtu.be/x"))
        assert results[-1] == "sum(short)"
        assert "Получение субтитров…" in results
        assert "Суммаризация…" in results

    def test_long_text_with_timed_segments(self, monkeypatch):
        """Long text with segments → timed map yields with timecodes."""
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))

        class FakeSummarizer:
            def __init__(self, *a):
                pass

            def summarize(self, text):
                return "s"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)

        from youtube import SubtitleResult, VideoInfo

        # Build timed segments exceeding MAP_REDUCE_THRESHOLD (4000 chars)
        segs = tuple((i * 10.0, "Слово. " * 100) for i in range(10))
        long_text = " ".join(t for _, t in segs)
        monkeypatch.setattr(
            "youtube.fetch_video_info",
            lambda url, **kw: VideoInfo(
                subtitles=SubtitleResult(text=long_text, language="ru", segments=segs),
                storyboard=None,
            ),
        )
        monkeypatch.setattr("youtube.validate_youtube_url", lambda u: u)

        results = list(app._default_youtube_summarize("https://youtu.be/x"))
        assert len(results) >= 4
        # Timecodes in output
        assert any("**[0:00]**" in r for r in results)
        assert results[-1] == "s"

    def test_long_text_untimed_fallback(self, monkeypatch):
        """Long text without segments → untimed chunk markers."""
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))

        class FakeSummarizer:
            def __init__(self, *a):
                pass

            def summarize(self, text):
                return "s"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)

        from youtube import SubtitleResult, VideoInfo

        long_text = "Слово. " * 1000
        monkeypatch.setattr(
            "youtube.fetch_video_info",
            lambda url, **kw: VideoInfo(
                subtitles=SubtitleResult(text=long_text, language="ru"),
                storyboard=None,
            ),
        )
        monkeypatch.setattr("youtube.validate_youtube_url", lambda u: u)

        results = list(app._default_youtube_summarize("https://youtu.be/x"))
        assert len(results) >= 4
        assert any("**[1/" in r for r in results)
        assert results[-1] == "s"

    def test_timed_with_storyboard_includes_thumbnail(self, monkeypatch):
        """Timed segments + storyboard → thumbnail HTML in output."""
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))

        class FakeSummarizer:
            def __init__(self, *a):
                pass

            def summarize(self, text):
                return "s"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)

        from youtube import StoryboardInfo, SubtitleResult, VideoInfo

        segs = tuple((i * 10.0, "Слово. " * 100) for i in range(10))
        long_text = " ".join(t for _, t in segs)
        sb = StoryboardInfo(
            fragments=(("https://cdn/M0.jpg", 200.0),),
            width=160,
            height=90,
            rows=10,
            columns=10,
            fps=0.5,
        )
        monkeypatch.setattr(
            "youtube.fetch_video_info",
            lambda url, **kw: VideoInfo(
                subtitles=SubtitleResult(text=long_text, language="ru", segments=segs),
                storyboard=sb,
            ),
        )
        monkeypatch.setattr("youtube.validate_youtube_url", lambda u: u)

        results = list(app._default_youtube_summarize("https://youtu.be/x"))
        assert any("<img" in r for r in results)
        assert any("https://cdn/M0.jpg" in r for r in results)

    def test_fallback_to_audio_when_no_subtitles(self, monkeypatch):
        """When subtitles unavailable, downloads audio and transcribes."""
        from config import OllamaConfig

        monkeypatch.setattr(app, "load_config", lambda: OllamaConfig("h", 1, "m"))

        class FakeSummarizer:
            def __init__(self, *a):
                pass

            def summarize(self, text):
                return "sum"

        monkeypatch.setattr(app, "Summarizer", FakeSummarizer)
        monkeypatch.setattr("youtube.validate_youtube_url", lambda u: u)
        monkeypatch.setattr(
            "youtube.fetch_video_info",
            lambda url, **kw: __import__("youtube").VideoInfo(
                subtitles=None, storyboard=None
            ),
        )
        monkeypatch.setattr(
            "youtube.download_audio", lambda url, **kw: "/tmp/audio.wav"
        )

        class FakeTranscriber:
            def transcribe(self, path):
                return TranscriptionResult(
                    text="transcribed", language="ru", duration=1.0
                )

        monkeypatch.setattr(app, "get_transcriber", lambda *a: FakeTranscriber())

        results = list(app._default_youtube_summarize("https://youtu.be/x"))
        assert results[-1] == "sum"
        assert "Субтитры не найдены. Загрузка аудио…" in results
        assert "Транскрибация аудио…" in results


# ---------- make_summarize_handler ----------


def _fake_summarize(text: str) -> str:
    return f"summary-of:{text}"


class TestMakeSummarizeHandler:
    def test_empty_string_returns_empty_transcript_message(self):
        summarize = make_summarize_handler(summarize_fn=_fake_summarize)
        assert _last(summarize("")) == EMPTY_TRANSCRIPT_MESSAGE

    def test_whitespace_only_returns_empty_transcript_message(self):
        summarize = make_summarize_handler(summarize_fn=_fake_summarize)
        assert _last(summarize("   ")) == EMPTY_TRANSCRIPT_MESSAGE

    def test_valid_transcript_returns_summary(self):
        summarize = make_summarize_handler(summarize_fn=_fake_summarize)
        assert _last(summarize("реальный текст")) == "summary-of:реальный текст"

    def test_summarize_fn_receives_full_transcript(self):
        seen: list[str] = []

        def spy(text: str) -> str:
            seen.append(text)
            return "ok"

        _last(make_summarize_handler(summarize_fn=spy)("некоторый текст"))
        assert seen == ["некоторый текст"]

    def test_exception_returns_ollama_unavailable(self):
        def boom(text: str) -> str:
            raise ConnectionError("refused")

        summarize = make_summarize_handler(summarize_fn=boom)
        assert _last(summarize("текст")) == OLLAMA_UNAVAILABLE

    def test_default_summarize_fn_used_when_not_injected(self, monkeypatch):
        called = {}

        def fake_default(text: str) -> str:
            called["text"] = text
            return "краткое содержание"

        monkeypatch.setattr(app, "_default_summarize", fake_default)
        summarize = app.make_summarize_handler()
        result = _last(summarize("входной текст"))
        assert result == "краткое содержание"
        assert called["text"] == "входной текст"

    def test_yields_progress_before_result(self):
        summarize = make_summarize_handler(summarize_fn=_fake_summarize)
        steps = list(summarize("текст"))
        assert len(steps) == 2
        assert steps[0] == "Суммаризация…"


# ---------- make_youtube_handler ----------


def _fake_youtube_summarize(url: str):
    yield f"yt-summary-of:{url}"


class TestMakeYoutubeHandler:
    def test_empty_string_returns_empty_url_message(self):
        handler = make_youtube_handler(youtube_summarize_fn=_fake_youtube_summarize)
        assert _last(handler("")) == EMPTY_URL_MESSAGE

    def test_whitespace_only_returns_empty_url_message(self):
        handler = make_youtube_handler(youtube_summarize_fn=_fake_youtube_summarize)
        assert _last(handler("   ")) == EMPTY_URL_MESSAGE

    def test_valid_url_returns_summary(self):
        handler = make_youtube_handler(youtube_summarize_fn=_fake_youtube_summarize)
        assert (
            _last(handler("https://youtu.be/abc"))
            == "yt-summary-of:https://youtu.be/abc"
        )

    def test_strips_url_whitespace(self):
        handler = make_youtube_handler(youtube_summarize_fn=_fake_youtube_summarize)
        assert (
            _last(handler("  https://youtu.be/abc  "))
            == "yt-summary-of:https://youtu.be/abc"
        )

    def test_value_error_returns_message(self):
        def raises_value(url: str):
            raise ValueError("Некорректная ссылка на YouTube.")
            yield  # noqa: RET503 — makes this a generator

        handler = make_youtube_handler(youtube_summarize_fn=raises_value)
        assert _last(handler("bad")) == "Некорректная ссылка на YouTube."

    def test_generic_exception_returns_youtube_error(self):
        def boom(url: str):
            raise RuntimeError("network failure")
            yield  # noqa: RET503 — makes this a generator

        handler = make_youtube_handler(youtube_summarize_fn=boom)
        assert _last(handler("https://youtu.be/abc")) == YOUTUBE_ERROR

    def test_fn_receives_stripped_url(self):
        seen: list[str] = []

        def spy(url: str):
            seen.append(url)
            yield "ok"

        _last(make_youtube_handler(youtube_summarize_fn=spy)("  url  "))
        assert seen == ["url"]

    def test_handler_is_generator(self):
        handler = make_youtube_handler(youtube_summarize_fn=_fake_youtube_summarize)
        import types

        assert isinstance(handler("x"), types.GeneratorType)

    def test_multiple_yields_from_fn(self):
        def multi(url: str):
            yield "step-1"
            yield "step-2"
            yield "final"

        handler = make_youtube_handler(youtube_summarize_fn=multi)
        results = list(handler("url"))
        assert results == ["step-1", "step-2", "final"]

    def test_default_fn_used_when_not_injected(self, monkeypatch):
        called = {}

        def fake_default(url: str):
            called["url"] = url
            yield "результат"

        monkeypatch.setattr(app, "_default_youtube_summarize", fake_default)
        handler = app.make_youtube_handler()
        result = _last(handler("https://youtu.be/abc"))
        assert result == "результат"
        assert called["url"] == "https://youtu.be/abc"


# ---------- build_app smoke test ----------


class TestBuildApp:
    def test_returns_gradio_blocks(self):
        import gradio as gr

        instance = build_app(
            transcribe_fn=_fake_transcribe,
            summarize_fn=_fake_summarize,
            youtube_summarize_fn=_fake_youtube_summarize,
        )
        assert isinstance(instance, gr.Blocks)

    def test_default_call_does_not_raise(self):
        # No arguments — uses module defaults.
        import gradio as gr

        instance = build_app()
        assert isinstance(instance, gr.Blocks)
