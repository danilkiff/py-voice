"""Tests for config.py."""

from __future__ import annotations

import pytest

from config import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, OllamaConfig, load_config

# ---------- OllamaConfig ----------


class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.host == DEFAULT_HOST
        assert cfg.port == DEFAULT_PORT
        assert cfg.model == DEFAULT_MODEL

    def test_base_url_default(self):
        cfg = OllamaConfig()
        assert cfg.base_url == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    def test_base_url_custom(self):
        cfg = OllamaConfig(host="10.0.0.1", port=12345)
        assert cfg.base_url == "http://10.0.0.1:12345"

    def test_is_frozen(self):
        import dataclasses

        cfg = OllamaConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            cfg.host = "other"  # type: ignore[misc]


# ---------- load_config ----------


class TestLoadConfig:
    def test_returns_defaults_when_no_env(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert load_config() == OllamaConfig()

    def test_env_host(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "myhost")
        assert load_config().host == "myhost"

    def test_env_port(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_PORT", "1234")
        assert load_config().port == 1234

    def test_env_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        assert load_config().model == "llama3"

    def test_env_port_as_string_is_coerced(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_PORT", "8080")
        assert load_config().port == 8080
