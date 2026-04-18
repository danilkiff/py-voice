---
status: accepted
date: 2025-05-01
---

# YouTube Subtitle Fast-Path with yt-dlp

## Context and Problem Statement

The application needs to summarize YouTube videos. A naive approach (download audio, transcribe, summarize) works but is very slow for long videos (tens of minutes). Most YouTube videos already have subtitles (manual or auto-generated) that can be fetched in seconds.

## Decision Drivers

* User experience: fast response for the common case
* Bandwidth: text is orders of magnitude smaller than audio
* Completeness: must still work when subtitles are unavailable
* Single library: avoid multiple YouTube client dependencies

## Considered Options

* **yt-dlp for both subtitle fetching (fast path) and audio download (fallback)**
* Always download audio and transcribe (simple, slow)
* youtube-transcript-api for subtitles + yt-dlp for audio (two libraries)
* Google YouTube Data API v3 (requires API key, limited subtitle access)

## Decision Outcome

Chosen option: **yt-dlp as sole YouTube client** with a two-phase pipeline:

1. **Fast path** — `fetch_subtitles(url)` calls `yt_dlp.YoutubeDL.extract_info(download=False)` to get subtitle metadata, then fetches the VTT/SRT content via httpx from the URL in the info dict. `parse_subtitle_text()` strips timestamps, indexes, and HTML tags to produce clean text. Preferred language order: Russian, then English.

2. **Slow path** — if no subtitles are found, `download_audio(url)` uses yt-dlp with `FFmpegExtractAudio` post-processor to download and convert to WAV, then the audio is transcribed via faster-whisper.

yt-dlp was chosen over alternatives because it is actively maintained, handles YouTube's evolving anti-scraping measures, and provides both subtitle metadata and audio download in a single library.

### Consequences

* Good, because videos with subtitles are summarized in seconds instead of minutes
* Good, because a single library handles both code paths
* Good, because injectable callables (`extract_info`, `downloader`) make both paths testable without hitting YouTube
* Bad, because two code paths (subtitles vs. audio) must be maintained
* Bad, because auto-generated subtitle quality varies
* Bad, because yt-dlp may break when YouTube changes its frontend (requires frequent updates)
* Bad, because FFmpeg must be installed as an external dependency for the audio path
