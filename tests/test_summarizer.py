"""Tests for summarizer.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from summarizer import SUMMARIZE_PROMPT, Summarizer

# ---------- test doubles ----------


@dataclass
class FakeResponse:
    _json: dict[str, Any] = field(default_factory=lambda: {"response": "  summary  "})
    status_code: int = 200
    raise_called: bool = False

    def raise_for_status(self) -> None:
        self.raise_called = True
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._json


@dataclass
class SpyPost:
    response: FakeResponse = field(default_factory=FakeResponse)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, url: str, body: dict[str, Any]) -> FakeResponse:
        self.calls.append((url, body))
        return self.response


# ---------- Summarizer ----------


class TestSummarizer:
    def test_calls_generate_endpoint(self):
        spy = SpyPost()
        Summarizer("http://localhost:11434", "gemma4:e4b", post=spy).summarize("txt")
        assert spy.calls[0][0] == "http://localhost:11434/api/generate"

    def test_custom_base_url(self):
        spy = SpyPost()
        Summarizer("http://host:9999", "m", post=spy).summarize("txt")
        assert spy.calls[0][0] == "http://host:9999/api/generate"

    def test_body_contains_model(self):
        spy = SpyPost()
        Summarizer("http://localhost:11434", "llama3", post=spy).summarize("txt")
        assert spy.calls[0][1]["model"] == "llama3"

    def test_body_stream_is_false(self):
        spy = SpyPost()
        Summarizer("http://localhost:11434", "m", post=spy).summarize("txt")
        assert spy.calls[0][1]["stream"] is False

    def test_prompt_template_rendered(self):
        spy = SpyPost()
        Summarizer("http://localhost:11434", "m", post=spy).summarize("мой текст")
        assert spy.calls[0][1]["prompt"] == SUMMARIZE_PROMPT.format(text="мой текст")

    def test_returns_stripped_response(self):
        spy = SpyPost(response=FakeResponse(_json={"response": "  результат  "}))
        result = Summarizer("http://localhost:11434", "m", post=spy).summarize("x")
        assert result == "результат"

    def test_raise_for_status_called(self):
        spy = SpyPost()
        Summarizer("http://localhost:11434", "m", post=spy).summarize("x")
        assert spy.response.raise_called

    def test_http_error_propagates(self):
        spy = SpyPost(response=FakeResponse(status_code=500))
        with pytest.raises(RuntimeError, match="HTTP 500"):
            Summarizer("http://localhost:11434", "m", post=spy).summarize("x")

    def test_default_post_used_when_not_injected(self):
        import summarizer

        s = Summarizer("http://localhost:11434", "m")
        assert s._post is summarizer._default_post
