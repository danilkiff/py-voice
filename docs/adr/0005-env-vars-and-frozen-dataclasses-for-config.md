---
status: accepted
date: 2025-05-01
---

# Environment Variables and Frozen Dataclasses for Configuration

## Context and Problem Statement

The application needs runtime configuration for the Ollama connection (host, port, model) and map-reduce parameters (chunk size, thresholds). The configuration mechanism must work seamlessly in Docker Compose, local development, and CI environments without risk of accidental mutation or credential leakage.

## Decision Drivers

* Docker-native: `docker-compose.yml` environment section is the primary config surface
* 12-factor app compliance: config stored in the environment, not in files
* Safety: no config files to accidentally commit with secrets
* Immutability: config values must not change after load

## Considered Options

* **Environment variables parsed into frozen dataclasses**
* YAML config file (previously used, then removed in commit 41300b4)
* JSON or TOML config files
* `.env` files with python-dotenv

## Decision Outcome

Chosen option: **environment variables into frozen dataclasses**.

`load_config()` in `config.py` reads `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL` from `os.environ` with sensible defaults and returns a `@dataclass(frozen=True)` `OllamaConfig`. The frozen constraint prevents accidental mutation after construction and makes the config safe to share across Gradio handler threads.

The same pattern is used for map-reduce tuning: `CHUNK_SIZE`, `MAP_REDUCE_THRESHOLD`, and `MAX_REDUCE_INPUT` are read from env vars at module load time in `map_reduce.py`.

Value objects throughout the codebase (`TranscriptionResult`, `SubtitleResult`, `OllamaConfig`) are all frozen dataclasses, establishing immutability as a project-wide convention.

### Consequences

* Good, because `docker-compose.yml` `environment:` section is the single source of truth
* Good, because frozen dataclasses raise `FrozenInstanceError` on accidental mutation
* Good, because no config files means no risk of committing secrets
* Good, because tests can override via `monkeypatch.setenv()` trivially
* Bad, because there is no multi-environment config (dev/staging/prod profiles)
* Bad, because env var values are unvalidated beyond basic type coercion (`int()`)
* Bad, because module-level `os.environ.get()` reads happen at import time, not lazily
