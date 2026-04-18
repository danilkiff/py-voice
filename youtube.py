"""YouTube subtitle fetching and audio downloading via yt-dlp.

The module is structured for testability:

* ``validate_youtube_url`` and ``parse_subtitle_text`` are pure functions.
* ``fetch_subtitles`` and ``download_audio`` take injectable callables,
  so tests can supply fakes instead of hitting real YouTube.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

YOUTUBE_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)
INVALID_URL_MESSAGE = "Некорректная ссылка на YouTube."

_VTT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}[:\.][\d.]+ --> \d{2}:\d{2}[:\.][\d.]+.*$", re.MULTILINE
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")
_SRT_INDEX_RE = re.compile(r"^\d+\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SubtitleResult:
    text: str
    language: str


# ---------------------------------------------------------------------------
# Injectable callable types
# ---------------------------------------------------------------------------
InfoExtractor = Callable[[str, dict[str, Any]], dict[str, Any]]
AudioDownloader = Callable[[str, dict[str, Any]], str]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def validate_youtube_url(url: str) -> str:
    """Return *url* unchanged if it looks like a valid YouTube link, else raise."""
    if not url or not YOUTUBE_URL_RE.search(url.strip()):
        raise ValueError(INVALID_URL_MESSAGE)
    return url.strip()


def parse_subtitle_text(raw: str) -> str:
    """Strip VTT/SRT timestamps, index lines and tags, returning plain text."""
    text = raw
    # Drop WEBVTT header block (header line + optional metadata lines)
    text = re.sub(
        r"^WEBVTT[^\n]*\n(?:[A-Z][a-zA-Z-]+:.*\n)*",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Drop NOTE blocks
    text = re.sub(r"^NOTE\s.*?(?:\n\n|\Z)", "", text, flags=re.MULTILINE | re.DOTALL)
    # Drop SRT numeric indexes
    text = _SRT_INDEX_RE.sub("", text)
    # Drop timestamp lines
    text = _VTT_TIMESTAMP_RE.sub("", text)
    # Drop HTML-like tags (e.g. <c>, </c>, <00:01:02.345>)
    text = _VTT_TAG_RE.sub("", text)
    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Default yt-dlp wrappers (lazy import)
# ---------------------------------------------------------------------------


def _default_extract_info(
    url: str, opts: dict[str, Any]
) -> dict[str, Any]:  # pragma: no cover
    import yt_dlp  # lazy

    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)  # type: ignore[return-value]


def _default_download(url: str, opts: dict[str, Any]) -> str:  # pragma: no cover
    import yt_dlp  # lazy

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SUBTITLE_LANGS = ("ru", "en")


def _fetch_subtitle_content(entry: dict[str, Any]) -> str:
    """Return subtitle text from *entry*, fetching from URL if needed."""
    raw = entry.get("data") or ""
    if raw:
        return raw
    sub_url = entry.get("url") or ""
    if not sub_url:
        return ""
    import httpx  # lazy

    resp = httpx.get(sub_url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


def fetch_subtitles(
    url: str,
    *,
    langs: tuple[str, ...] = _SUBTITLE_LANGS,
    extract_info: InfoExtractor | None = None,
) -> SubtitleResult | None:
    """Try to get subtitles for *url*. Return ``None`` if none available."""
    extractor = extract_info or _default_extract_info
    opts: dict[str, Any] = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "skip_download": True,
        "quiet": True,
    }
    info = extractor(url, opts)

    for source_key in ("subtitles", "automatic_captions"):
        subs = info.get(source_key) or {}
        for lang in langs:
            entries = subs.get(lang)
            if not entries:
                continue
            # Prefer the "vtt" or "srv1" format, fall back to the first entry.
            for entry in entries:
                if entry.get("ext") in ("vtt", "srv1", "srt"):
                    raw = _fetch_subtitle_content(entry)
                    if raw:
                        return SubtitleResult(
                            text=parse_subtitle_text(raw),
                            language=lang,
                        )
            # Fallback: first entry with any content
            raw = _fetch_subtitle_content(entries[0])
            if raw:
                return SubtitleResult(
                    text=parse_subtitle_text(raw),
                    language=lang,
                )
    return None


def download_audio(
    url: str,
    *,
    output_dir: Path | None = None,
    downloader: AudioDownloader | None = None,
) -> Path:
    """Download audio track and return path to the resulting file."""
    dl = downloader or _default_download
    target = output_dir or Path(tempfile.mkdtemp())
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(target / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            },
        ],
        "quiet": True,
    }
    result_path = dl(url, opts)
    # yt-dlp returns the pre-postprocessor filename; the actual file has .wav
    wav_path = Path(result_path).with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    return Path(result_path)
