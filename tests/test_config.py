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
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg == OllamaConfig()

    def test_full_yaml(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "ollama:\n  host: 192.168.1.1\n  port: 12345\n  model: llama3\n",
            encoding="utf-8",
        )
        cfg = load_config(f)
        assert cfg.host == "192.168.1.1"
        assert cfg.port == 12345
        assert cfg.model == "llama3"

    def test_partial_yaml_fills_defaults(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("ollama:\n  host: myserver\n", encoding="utf-8")
        cfg = load_config(f)
        assert cfg.host == "myserver"
        assert cfg.port == DEFAULT_PORT
        assert cfg.model == DEFAULT_MODEL

    def test_empty_file_returns_defaults(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("", encoding="utf-8")
        assert load_config(f) == OllamaConfig()

    def test_no_ollama_section_returns_defaults(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("other:\n  key: value\n", encoding="utf-8")
        assert load_config(f) == OllamaConfig()

    def test_port_as_string_is_coerced(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("ollama:\n  port: '8080'\n", encoding="utf-8")
        assert load_config(f).port == 8080

    def test_accepts_str_path(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("ollama:\n  host: strhost\n", encoding="utf-8")
        cfg = load_config(str(f))
        assert cfg.host == "strhost"
