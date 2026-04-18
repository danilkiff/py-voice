---
status: accepted
date: 2025-05-01
---

# Docker Compose for Development Stack

## Context and Problem Statement

The application requires two services to run: the py-voice web app and an Ollama LLM server with a pre-pulled model. Developers need a single command to bring up the full stack with correct networking, health checks, and startup ordering.

## Decision Drivers

* One-command startup: `docker compose up --build`
* Reproducible across developer machines (macOS, Linux)
* Ollama must be healthy and model pulled before the app starts
* Optional NVIDIA GPU passthrough for faster inference

## Considered Options

* **Docker Compose with health checks and sidecar model pull**
* Shell scripts with manual `docker run` commands
* Kubernetes (Helm chart)
* systemd units (Linux-only)

## Decision Outcome

Chosen option: **Docker Compose** with three services:

1. **ollama** — LLM inference server with a health check (`ollama list`, 10 s interval) and a persistent volume (`ollama_data`) for cached models. GPU support is available via a commented NVIDIA runtime section.

2. **ollama-pull** — one-shot sidecar that runs `ollama pull gemma4:e4b` after the ollama service is healthy (`depends_on: ollama: condition: service_healthy`). Exits after model is ready.

3. **py-voice** — the web app, built from `Dockerfile`. Depends on `ollama-pull` with `condition: service_completed_successfully`, ensuring the model is available before the app accepts requests. Environment variable `OLLAMA_HOST=ollama` points the app to the Ollama container.

The **Dockerfile** uses a dependency-layer caching strategy: `pyproject.toml` + `uv.lock` are copied and `uv sync --frozen` runs before the source code is copied, so dependency installation is cached across code-only rebuilds.

### Consequences

* Good, because `docker compose up --build` starts everything with correct ordering
* Good, because health checks prevent race conditions (app doesn't start before model is ready)
* Good, because dependency layer caching speeds up rebuilds when only code changes
* Good, because the volume persists the Ollama model across restarts
* Bad, because GPU support requires uncommenting a section (easy to miss)
* Bad, because the health check adds ~10 s startup delay even when Ollama is ready instantly
* Bad, because the model name is hardcoded in the sidecar command (must be kept in sync with `OLLAMA_MODEL`)
