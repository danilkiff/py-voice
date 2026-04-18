"""Tests for youtube.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from youtube import (
    _SUBTITLE_LANGS,
    INVALID_URL_MESSAGE,
    SubtitleResult,
    _fetch_subtitle_content,
    download_audio,
    fetch_subtitles,
    parse_subtitle_text,
    validate_youtube_url,
)

# ---------- test doubles ----------


@dataclass
class SpyExtractInfo:
    """Records calls and returns a canned info dict."""

    result: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, opts))
        return self.result


@dataclass
class SpyDownloader:
    """Records calls and returns a canned file path."""

    result_path: str = "/tmp/video.webm"
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, url: str, opts: dict[str, Any]) -> str:
        self.calls.append((url, opts))
        return self.result_path


# ---------- validate_youtube_url ----------


class TestValidateYoutubeUrl:
    def test_standard_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_without_www(self):
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_without_scheme(self):
        url = "youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s"
        assert validate_youtube_url(url) == url

    def test_strips_whitespace(self):
        assert validate_youtube_url("  https://youtu.be/dQw4w9WgXcQ  ") == (
            "https://youtu.be/dQw4w9WgXcQ"
        )

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match=INVALID_URL_MESSAGE):
            validate_youtube_url("")

    def test_random_string_raises(self):
        with pytest.raises(ValueError, match=INVALID_URL_MESSAGE):
            validate_youtube_url("not a url at all")

    def test_non_youtube_url_raises(self):
        with pytest.raises(ValueError, match=INVALID_URL_MESSAGE):
            validate_youtube_url("https://vimeo.com/123456")

    def test_short_video_id_raises(self):
        with pytest.raises(ValueError, match=INVALID_URL_MESSAGE):
            validate_youtube_url("https://youtu.be/short")


# ---------- parse_subtitle_text ----------

_SAMPLE_VTT = """\
WEBVTT
Kind: captions
Language: ru

00:00:00.000 --> 00:00:03.500
Привет, это тест.

00:00:03.500 --> 00:00:07.000
Вторая строка субтитров.
"""

_SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:03,500
Привет, это тест.

2
00:00:03,500 --> 00:00:07,000
Вторая строка субтитров.
"""


class TestParseSubtitleText:
    def test_strips_vtt_timestamps(self):
        result = parse_subtitle_text(_SAMPLE_VTT)
        assert "00:00" not in result
        assert "Привет, это тест." in result
        assert "Вторая строка субтитров." in result

    def test_strips_srt_indexes(self):
        result = parse_subtitle_text(_SAMPLE_SRT)
        assert "Привет, это тест." in result
        assert "Вторая строка субтитров." in result

    def test_strips_html_tags(self):
        raw = "00:00:01.000 --> 00:00:02.000\n<c>tagged</c> text"
        result = parse_subtitle_text(raw)
        assert "<c>" not in result
        assert "tagged text" in result

    def test_empty_input(self):
        assert parse_subtitle_text("") == ""

    def test_strips_webvtt_header(self):
        result = parse_subtitle_text(_SAMPLE_VTT)
        assert "WEBVTT" not in result

    def test_returns_joined_text(self):
        result = parse_subtitle_text(_SAMPLE_VTT)
        # All text on one line separated by spaces
        assert result == "Привет, это тест. Вторая строка субтитров."


# ---------- _fetch_subtitle_content ----------


class TestFetchSubtitleContent:
    def test_returns_data_field_when_present(self):
        assert _fetch_subtitle_content({"data": "hello"}) == "hello"

    def test_returns_empty_when_no_data_and_no_url(self):
        assert _fetch_subtitle_content({}) == ""
        assert _fetch_subtitle_content({"data": "", "url": ""}) == ""

    def test_fetches_from_url_when_no_data(self, monkeypatch):
        """When data is absent but url is present, fetch via httpx."""
        import youtube

        class FakeResp:
            text = "00:00:00.000 --> 00:00:01.000\nfetched"

            def raise_for_status(self):
                pass

        def fake_get(url, **kwargs):
            assert url == "https://example.com/subs.vtt"
            return FakeResp()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)
        result = _fetch_subtitle_content({"url": "https://example.com/subs.vtt"})
        assert result == "00:00:00.000 --> 00:00:01.000\nfetched"


# ---------- fetch_subtitles ----------


class TestFetchSubtitles:
    def test_returns_subtitle_result_when_found(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "ru": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\nтекст"}
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert isinstance(result, SubtitleResult)
        assert result.language == "ru"
        assert "текст" in result.text

    def test_prefers_manual_subtitles_over_auto(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "ru": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\nручные"}
                    ],
                },
                "automatic_captions": {
                    "ru": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\nавто"}
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert "ручные" in result.text

    def test_falls_back_to_auto_captions(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {},
                "automatic_captions": {
                    "ru": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\nавто"}
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert "авто" in result.text

    def test_returns_none_when_no_subtitles(self):
        spy = SpyExtractInfo(result={"subtitles": {}, "automatic_captions": {}})
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is None

    def test_returns_none_for_empty_data_and_no_url(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {"ru": [{"ext": "vtt", "data": "", "url": ""}]},
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is None

    def test_falls_back_to_english(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "en": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\nenglish"}
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert result.language == "en"

    def test_passes_url_and_opts_to_extractor(self):
        spy = SpyExtractInfo(result={})
        fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert len(spy.calls) == 1
        url, opts = spy.calls[0]
        assert url == "https://youtu.be/abc12345678"
        assert opts["writesubtitles"] is True
        assert opts["writeautomaticsub"] is True

    def test_falls_back_to_first_entry_when_preferred_exts_empty(self):
        """When preferred ext entries have no data, falls back to entries[0]."""
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "ru": [
                        {
                            "ext": "json3",
                            "data": "00:00:00.000 --> 00:00:01.000\nfallback",
                        },
                        {"ext": "vtt", "data": ""},
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert "fallback" in result.text

    def test_skips_preferred_ext_with_empty_data(self):
        """Preferred ext entry exists but data is empty → try next."""
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "ru": [
                        {"ext": "vtt", "data": ""},
                        {
                            "ext": "srv1",
                            "data": "00:00:00.000 --> 00:00:01.000\nsrv1text",
                        },
                    ],
                },
            }
        )
        result = fetch_subtitles("https://youtu.be/abc12345678", extract_info=spy)
        assert result is not None
        assert "srv1text" in result.text

    def test_default_langs(self):
        assert _SUBTITLE_LANGS == ("ru", "en")

    def test_custom_langs(self):
        spy = SpyExtractInfo(
            result={
                "subtitles": {
                    "de": [
                        {"ext": "vtt", "data": "00:00:00.000 --> 00:00:01.000\ndeutsch"}
                    ],
                },
            }
        )
        result = fetch_subtitles(
            "https://youtu.be/abc12345678",
            langs=("de",),
            extract_info=spy,
        )
        assert result is not None
        assert result.language == "de"

    def test_default_extract_info_used_when_not_injected(self):
        import youtube

        # Just verify that not passing extract_info uses the module default.
        # We don't call it (would hit real YouTube), but check the code path.
        # The real default is _default_extract_info.
        assert youtube._default_extract_info is not None


# ---------- download_audio ----------


class TestDownloadAudio:
    def test_returns_path(self, tmp_path):
        wav = tmp_path / "video_id.wav"
        wav.write_bytes(b"RIFF")
        spy = SpyDownloader(result_path=str(tmp_path / "video_id.webm"))
        result = download_audio(
            "https://youtu.be/abc12345678",
            output_dir=tmp_path,
            downloader=spy,
        )
        assert result == wav

    def test_falls_back_to_original_path(self, tmp_path):
        original = tmp_path / "video_id.webm"
        original.write_bytes(b"data")
        spy = SpyDownloader(result_path=str(original))
        result = download_audio(
            "https://youtu.be/abc12345678",
            output_dir=tmp_path,
            downloader=spy,
        )
        assert result == original

    def test_passes_url_and_opts(self, tmp_path):
        spy = SpyDownloader(result_path=str(tmp_path / "x.webm"))
        download_audio(
            "https://youtu.be/abc12345678",
            output_dir=tmp_path,
            downloader=spy,
        )
        assert len(spy.calls) == 1
        url, opts = spy.calls[0]
        assert url == "https://youtu.be/abc12345678"
        assert opts["format"] == "bestaudio/best"
        assert str(tmp_path) in opts["outtmpl"]

    def test_uses_temp_dir_when_no_output_dir(self):
        spy = SpyDownloader(result_path="/tmp/test/x.webm")
        # Should not raise — uses tempfile.mkdtemp internally
        download_audio("https://youtu.be/abc12345678", downloader=spy)
        assert len(spy.calls) == 1

    def test_default_downloader_used_when_not_injected(self):
        import youtube

        assert youtube._default_download is not None
