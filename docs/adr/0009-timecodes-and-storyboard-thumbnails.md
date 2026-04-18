---
status: accepted
date: 2026-04-18
---

# Timecodes and Storyboard Thumbnails for YouTube Summaries

## Context and Problem Statement

YouTube video summaries display chunk markers like `**[1/5]**` with no indication of where in the video each chunk comes from. Users cannot navigate to the relevant part of the video from the summary. Meanwhile, VTT/SRT subtitles contain timestamps that the pipeline strips, and yt-dlp returns storyboard sprite sheet metadata that the pipeline ignores.

## Decision Drivers

* User orientation: summaries should anchor to video timeline
* No additional API calls: reuse data already available from `extract_info()`
* No new binary dependencies: avoid requiring Pillow or other image libraries
* Backward compatibility: existing untimed and audio-fallback paths must continue working
* Minimal invasiveness: avoid modifying `chunk_text()` and `map_reduce_summarize()`

## Considered Options

* **CSS sprite cropping with a parallel timed-segment path**
* Server-side sprite cropping with Pillow (PIL)
* YouTube oEmbed / Data API v3 thumbnail endpoints
* Modifying `chunk_text()` to accept and return timestamp metadata

## Decision Outcome

Chosen option: **CSS sprite cropping with a parallel timed-segment path**, because it requires no new dependencies, reuses existing yt-dlp data, and keeps the core chunking/map-reduce pipeline unchanged.

### Timed subtitle parsing

`parse_subtitle_timed(raw)` parses VTT/SRT into `(start_seconds, text)` pairs, preserving timestamp data that `parse_subtitle_text()` strips. Both functions coexist — the original remains for backward compatibility with `fetch_subtitles()`.

### Single extract_info call

`fetch_video_info(url)` replaces `fetch_subtitles(url)` in the YouTube pipeline. It calls `extract_info()` once and returns a `VideoInfo` dataclass containing both `SubtitleResult` (now extended with a `segments` field) and `StoryboardInfo`. The subtitle selection logic is extracted into `_find_best_subtitle()`, shared by both `fetch_subtitles()` and `fetch_video_info()`.

### Segment-based chunking

`_group_timed_segments()` groups subtitle segments by cumulative character length instead of using `chunk_text()`. Each group retains the timestamp of its first segment. This avoids modifying `chunk_text()` or threading metadata through `map_reduce_summarize()`.

### Storyboard thumbnails via CSS

`extract_storyboard(info)` reads the `formats` array for entries with `format_note == "storyboard"` and returns sprite sheet metadata (fragment URLs, grid dimensions, fps). `thumbnail_url_for_time()` computes which sprite sheet and pixel offset corresponds to a given timestamp. The thumbnail is rendered as an `<img>` tag with `object-fit:none` and `object-position` CSS — the browser crops the correct frame from the sprite sheet, with no server-side image processing.

### Output format

Map phase output changes from `**[1/5]** summary` to `**[0:32]** summary` with an optional `<img>` thumbnail below each chunk summary.

### Consequences

* Good, because summaries now show timecodes that anchor each chunk to the video timeline
* Good, because storyboard thumbnails provide visual context with zero additional HTTP requests (sprites are already on YouTube CDN)
* Good, because no new dependencies are required (CSS handles sprite cropping)
* Good, because `chunk_text()` and `map_reduce_summarize()` are untouched
* Good, because `fetch_subtitles()` continues to work unchanged for other callers
* Bad, because the timed path only works with subtitles — the audio transcription fallback has no timecodes (future work: extend `_Segment` protocol with `.start`)
* Bad, because storyboard sprite URLs contain authentication tokens that expire after hours
* Bad, because Gradio's Markdown HTML sanitizer may strip CSS properties in future versions
* Bad, because subtitle timestamps from auto-generated captions can be imprecise
