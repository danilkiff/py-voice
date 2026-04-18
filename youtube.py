"""YouTube subtitle fetching and audio downloading via yt-dlp.

The module is structured for testability:

* ``validate_youtube_url``, ``parse_subtitle_text`` and ``parse_subtitle_timed``
  are pure functions.
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
_VTT_TIMED_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->", re.MULTILINE)
_VTT_TAG_RE = re.compile(r"<[^>]+>")
_SRT_INDEX_RE = re.compile(r"^\d+\s*$", re.MULTILINE)


@dataclass(frozen=True)
class SubtitleResult:
    text: str
    language: str
    segments: tuple[tuple[float, str], ...] = ()


@dataclass(frozen=True)
class StoryboardInfo:
    """YouTube storyboard sprite metadata for thumbnail extraction."""

    fragments: tuple[tuple[str, float], ...]  # (url, duration_seconds)
    width: int
    height: int
    rows: int
    columns: int
    fps: float


@dataclass(frozen=True)
class VideoInfo:
    """Combined result of subtitle + storyboard extraction."""

    subtitles: SubtitleResult | None
    storyboard: StoryboardInfo | None


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


def parse_subtitle_timed(raw: str) -> tuple[tuple[float, str], ...]:
    """Parse VTT/SRT into (start_seconds, text) pairs, preserving timestamps."""
    text = raw
    # Drop WEBVTT header block
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

    segments: list[tuple[float, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _VTT_TIMED_RE.match(lines[i])
        if m:
            h, mn, s, ms = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
            )
            start = h * 3600 + mn * 60 + s + ms / 1000.0
            i += 1
            cue_lines: list[str] = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not _VTT_TIMED_RE.match(lines[i])
            ):
                cleaned = _VTT_TAG_RE.sub("", lines[i]).strip()
                if cleaned:
                    cue_lines.append(cleaned)
                i += 1
            if cue_lines:
                segments.append((start, " ".join(cue_lines)))
        else:
            i += 1
    return tuple(segments)


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


def extract_storyboard(info: dict[str, Any]) -> StoryboardInfo | None:
    """Extract storyboard sprite metadata from yt-dlp info dict."""
    best: dict[str, Any] | None = None
    for fmt in info.get("formats") or []:
        if fmt.get("format_note") != "storyboard":
            continue
        if best is None or (fmt.get("width", 0) * fmt.get("height", 0)) > (
            best.get("width", 0) * best.get("height", 0)
        ):
            best = fmt
    if best is None:
        return None
    fragments = tuple((f["url"], f["duration"]) for f in best.get("fragments") or [])
    if not fragments:
        return None
    return StoryboardInfo(
        fragments=fragments,
        width=best.get("width", 0),
        height=best.get("height", 0),
        rows=best.get("rows", 1),
        columns=best.get("columns", 1),
        fps=best.get("fps", 0.0),
    )


def thumbnail_url_for_time(
    storyboard: StoryboardInfo, seconds: float
) -> tuple[str, int, int]:
    """Return (sprite_url, x_offset_px, y_offset_px) for *seconds*."""
    if storyboard.fps <= 0:
        return storyboard.fragments[0][0], 0, 0

    thumbs_per_sheet = storyboard.rows * storyboard.columns
    # Find which fragment contains this timestamp
    elapsed = 0.0
    frag_idx = 0
    for idx, (_url, duration) in enumerate(storyboard.fragments):  # pragma: no branch
        if elapsed + duration > seconds or idx == len(storyboard.fragments) - 1:
            frag_idx = idx
            break
        elapsed += duration

    time_in_frag = max(0.0, seconds - elapsed)
    thumb_idx = int(time_in_frag * storyboard.fps)
    thumb_idx = min(thumb_idx, thumbs_per_sheet - 1)

    col = thumb_idx % storyboard.columns
    row = thumb_idx // storyboard.columns
    return (
        storyboard.fragments[frag_idx][0],
        col * storyboard.width,
        row * storyboard.height,
    )


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


def _find_best_subtitle(
    info: dict[str, Any], langs: tuple[str, ...]
) -> tuple[str, str] | None:
    """Return (raw_subtitle_text, language) from *info*, or ``None``."""
    for source_key in ("subtitles", "automatic_captions"):
        subs = info.get(source_key) or {}
        for lang in langs:
            entries = subs.get(lang)
            if not entries:
                continue
            for entry in entries:
                if entry.get("ext") in ("vtt", "srv1", "srt"):
                    raw = _fetch_subtitle_content(entry)
                    if raw:
                        return raw, lang
            raw = _fetch_subtitle_content(entries[0])
            if raw:
                return raw, lang
    return None


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
    result = _find_best_subtitle(info, langs)
    if result is None:
        return None
    raw, lang = result
    return SubtitleResult(text=parse_subtitle_text(raw), language=lang)


def fetch_video_info(
    url: str,
    *,
    langs: tuple[str, ...] = _SUBTITLE_LANGS,
    extract_info: InfoExtractor | None = None,
) -> VideoInfo:
    """Fetch subtitles (with timing) and storyboard in a single extract_info call."""
    extractor = extract_info or _default_extract_info
    opts: dict[str, Any] = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "skip_download": True,
        "quiet": True,
    }
    info = extractor(url, opts)

    subtitles: SubtitleResult | None = None
    result = _find_best_subtitle(info, langs)
    if result is not None:
        raw, lang = result
        subtitles = SubtitleResult(
            text=parse_subtitle_text(raw),
            language=lang,
            segments=parse_subtitle_timed(raw),
        )

    return VideoInfo(subtitles=subtitles, storyboard=extract_storyboard(info))


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
