"""Load YAML configuration for the application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.yaml")
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


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> OllamaConfig:
    """Return OllamaConfig from a YAML file. Missing file → defaults."""
    import yaml  # lazy: keeps import cheap when yaml is unused

    p = Path(path)
    if not p.is_file():
        return OllamaConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ollama = data.get("ollama") or {}
    return OllamaConfig(
        host=str(ollama.get("host", DEFAULT_HOST)),
        port=int(ollama.get("port", DEFAULT_PORT)),
        model=str(ollama.get("model", DEFAULT_MODEL)),
    )
