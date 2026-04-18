"""Summarize text via Ollama REST API.

The module is structured for testability:

* ``SUMMARIZE_PROMPT`` is a pure constant — easy to assert against in tests.
* ``Summarizer`` takes an injectable ``post`` callable, so tests can supply
  fakes instead of making real HTTP requests.
* ``_default_post`` is the only place that imports and uses ``httpx``.
"""

from __future__ import annotations

from typing import Any, Callable

SUMMARIZE_PROMPT = (
    "Выдели 2–3 ключевых тезиса из текста ниже. "
    "Формат: маркированный список, каждый пункт — одно короткое предложение. "
    "Без вводных слов, пояснений и выводов. Язык — русский.\n\n{text}"
)

# A callable with the same signature as the injectable hook.
HttpPost = Callable[[str, dict[str, Any]], Any]


def _default_post(url: str, body: dict[str, Any]) -> Any:  # pragma: no cover
    import httpx  # lazy: keeps import cheap when httpx is unused

    with httpx.Client(timeout=120.0) as client:
        return client.post(url, json=body)


class Summarizer:
    """Calls Ollama /api/generate to produce a Russian summary."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        post: HttpPost | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._post: HttpPost = post or _default_post

    def summarize(self, text: str) -> str:
        resp = self._post(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": SUMMARIZE_PROMPT.format(text=text),
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
