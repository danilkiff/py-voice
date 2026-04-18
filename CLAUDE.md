# CLAUDE.md

## Project

Russian audio transcription + YouTube video summarization. Gradio UI, faster-whisper STT, Ollama LLM.

## Commands

```bash
uv sync              # install deps
uv run pytest        # test (also runs on pre-commit)
uv run main.py       # start app on 0.0.0.0:7860
docker compose up --build  # full stack with Ollama
```

## Architecture

```
main.py          → entry point, launches Gradio
app.py           → UI + handler factories + YouTube pipeline (_default_youtube_summarize)
transcriber.py   → faster-whisper wrapper, lazy model loading
summarizer.py    → Ollama /api/generate client
youtube.py       → yt-dlp subtitle/storyboard fetch + audio download (VideoInfo, SubtitleResult, StoryboardInfo)
map_reduce.py    → chunk_text + recursive map-reduce summarize
config.py        → OllamaConfig from env vars
device.py        → CUDA auto-detection via CTranslate2
```

## Patterns — follow these

- **Dependency injection** via callable parameters, not mocks. Every class/factory accepts optional injectable functions (`loader=`, `post=`, `extract_info=`). Defaults are private `_default_*` functions with lazy imports.
- **Protocol typing** for injectable interfaces (`_Model`, `HttpPost`, `InfoExtractor`).
- **Frozen dataclasses** for value objects (`TranscriptionResult`, `SubtitleResult`, `OllamaConfig`).
- **Dataclass fakes** in tests (`FakeModel`, `SpyPost`, `SpyExtractInfo`). No `unittest.mock.Mock()`.
- **Lazy imports** — heavy libs (`faster_whisper`, `yt_dlp`, `httpx`) imported inside function bodies, not at module level.
- **Handler factories** — `make_*_handler(fn=None)` pattern decouples Gradio UI from business logic. All handlers are generators (yield progress, then result).
- **Streaming reduce** — `_streaming_reduce` in `app.py` mirrors `map_reduce_summarize` logic but yields per-call progress. Keep both in sync.
- **Timed YouTube path** — when VTT/SRT subtitles are available, `fetch_video_info` returns `VideoInfo` with timed segments + storyboard metadata. Chunks are grouped by `_group_timed_segments` (not `chunk_text`), output includes timecodes and CSS-cropped storyboard thumbnails.
- **Env var config** — no config files. `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`, `CHUNK_SIZE`, `MAP_REDUCE_THRESHOLD`, `MAX_REDUCE_INPUT`.

## Testing

- 100% branch coverage required. Pre-commit hook runs ruff + pytest.
- Thin I/O wrappers (`_default_post`, `_default_extract_info`, `_default_download`) are `# pragma: no cover`.
- Protocol method stubs excluded via `exclude_lines` in `pyproject.toml`.
- New modules must be added to `--cov=` in `pyproject.toml` addopts and `[tool.coverage.run] source`.

## Code style

- `uv run ruff check --fix . && uv run ruff format .`
- Russian UI strings, English code/comments/commits.
- Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`).

## ADRs

Architectural decisions documented in `docs/adr/` (MADR format, English).
