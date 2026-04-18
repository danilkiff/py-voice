---
status: accepted
date: 2025-05-01
---

# Use Gradio as Web UI Framework

## Context and Problem Statement

The application needs a web interface for uploading audio files, selecting models, viewing transcription results, and triggering summarization. The UI should be simple to build and maintain, with built-in support for audio input and file download.

## Decision Drivers

* Minimal boilerplate for ML-oriented UIs
* Built-in audio upload and file download components
* Quick iteration without frontend build tooling
* Functional composition for testable handler logic

## Considered Options

* **Gradio** (declarative ML UI framework)
* Flask + HTML templates (traditional web framework)
* FastAPI + React (REST API with separate SPA frontend)
* Streamlit (alternative ML UI framework)

## Decision Outcome

Chosen option: **Gradio**, because its declarative API produces a complete two-tab UI (`app.py:build_app`) in ~100 lines with native audio upload, file download, and text display components.

The handler factory pattern (`make_run_handler`, `make_summarize_handler`, `make_youtube_handler`) decouples UI wiring from business logic, making handlers independently testable via dependency injection.

### Consequences

* Good, because the entire UI is defined in a single Python file with no frontend build step
* Good, because `gr.Audio`, `gr.File`, `gr.Tabs` components match the exact use case
* Good, because handler factories enable testing without rendering the UI
* Bad, because styling customization is limited compared to a traditional web framework
* Bad, because there is no programmatic REST API (inference is UI-only)
* Bad, because Gradio v5+ may introduce breaking API changes
