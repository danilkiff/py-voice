"""Load configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "gemma4:e4b"


@dataclass(frozen=True)
class OllamaConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model: str = DEFAULT_MODEL

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_config() -> OllamaConfig:
    """Return OllamaConfig from environment variables. Missing var → default."""
    return OllamaConfig(
        host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
        port=int(os.environ.get("OLLAMA_PORT", DEFAULT_PORT)),
        model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
    )
