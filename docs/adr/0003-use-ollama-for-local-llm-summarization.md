---
status: accepted
date: 2025-05-01
---

# Use Ollama for Local LLM Summarization

## Context and Problem Statement

Transcribed text needs to be summarized into a concise Russian-language overview. The summarization backend must be self-hosted, support Russian well, and integrate cleanly with the existing Docker Compose deployment.

## Decision Drivers

* Privacy: transcribed text must not leave the local network
* Cost: no per-token API charges
* Deployment: must run as a Docker service alongside the app
* Language: must handle Russian-language prompts and output

## Considered Options

* **Ollama** (local LLM inference server with REST API)
* OpenAI GPT API (cloud, per-token billing)
* Anthropic Claude API (cloud, per-token billing)
* Local llama.cpp (no HTTP API out of the box)

## Decision Outcome

Chosen option: **Ollama**, because it provides a Docker-native REST API (`/api/generate`) that the app calls via httpx, with no API keys or billing required.

The default model (`gemma4:e4b`, configurable via `OLLAMA_MODEL` env var) handles Russian summarization adequately. The `Summarizer` class in `summarizer.py` is a thin HTTP client with an injectable `post` callable for testability.

Docker Compose orchestrates Ollama startup, health checks, and model pre-pulling via a sidecar service (`ollama-pull`), so `docker compose up` works out of the box.

### Consequences

* Good, because summarization is free and private
* Good, because Docker Compose handles Ollama lifecycle automatically
* Good, because the model is swappable via a single env var
* Bad, because Ollama requires a separate container with 8 GB+ RAM
* Bad, because local inference is slower than cached cloud APIs (120 s timeout configured)
* Bad, because model quality depends on available hardware (quantized models on CPU)
