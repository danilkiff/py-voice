---
status: accepted
date: 2025-05-01
---

# Dependency Injection with Protocol Typing for Testability

## Context and Problem Statement

The application depends on heavy external systems (faster-whisper models, Ollama HTTP API, yt-dlp/YouTube). Tests must run fast, without downloading models or hitting real services, while keeping the code type-safe and explicit about its dependencies.

## Decision Drivers

* Tests must run in < 30 s without network or GPU
* Dependencies must be visible in function signatures, not hidden behind mocks
* Static type checkers (mypy, Pyright) must be able to verify call sites
* Heavy libraries (faster-whisper, yt-dlp, httpx) must not slow down module import

## Considered Options

* **Constructor-injectable callables with Protocol types**
* `unittest.mock.patch()` / `monkeypatch` everywhere
* Abstract base classes (ABC) with concrete implementations
* pytest-mock fixtures

## Decision Outcome

Chosen option: **injectable callables with Protocol types**, applied consistently across all modules:

1. **Protocol types** define the expected callable shape (`_Model`, `HttpPost`, `InfoExtractor`, `AudioDownloader`) so type checkers verify fakes match real implementations.

2. **Constructor injection** — every class and handler factory accepts an optional callable parameter (e.g., `Transcriber(loader=...)`, `Summarizer(post=...)`, `make_youtube_handler(youtube_summarize_fn=...)`). The default resolves to a private `_default_*` function.

3. **Lazy imports** — default implementations import heavy libraries inside the function body (`import yt_dlp`, `import httpx`, `from faster_whisper import WhisperModel`), keeping module-level imports cheap. Tests substitute fakes at the injection point, so heavy libraries are never loaded.

4. **Dataclass fakes** — test doubles are plain `@dataclass` classes (`FakeModel`, `SpyPost`, `SpyExtractInfo`) that record calls and return canned data. No `unittest.mock.Mock()` is used anywhere.

### Consequences

* Good, because tests run in < 30 s with zero network or GPU access
* Good, because function signatures document all dependencies explicitly
* Good, because Protocol typing catches interface mismatches at type-check time
* Good, because lazy imports keep `import transcriber` instant (no 3 GB model load)
* Bad, because every dependency needs a Protocol definition (more boilerplate)
* Bad, because import errors are delayed to first use rather than module load
* Bad, because `_default_*` functions add an indirection layer to follow
