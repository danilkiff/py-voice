"""Map-reduce summarization for long texts.

The module is structured for testability:

* ``chunk_text`` is a pure function — easy to unit-test with plain strings.
* ``map_reduce_summarize`` takes an injectable ``summarize_fn`` so tests can
  supply a spy instead of calling a real LLM.

Context-window considerations
-----------------------------
Local models (e.g. qwen3:8b) typically have an 8 K-token context window,
which is roughly 24 K Cyrillic characters.  The defaults below are tuned
for that constraint:

* ``DEFAULT_CHUNK_SIZE = 3000`` chars — fits comfortably with the prompt.
* ``MAX_REDUCE_INPUT = 20000`` chars — keeps the reduce step within 8 K
  tokens.  If the combined map summaries exceed this, the reduce is run
  recursively (chunk → summarize → reduce again) until it fits.

For models with a larger context (32 K+) these values can be raised via
environment variables ``CHUNK_SIZE``, ``MAP_REDUCE_THRESHOLD``, and
``MAX_REDUCE_INPUT``.
"""

from __future__ import annotations

import os
from typing import Callable

DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "3000"))
DEFAULT_OVERLAP = 200  # character overlap between adjacent chunks
MAP_REDUCE_THRESHOLD = int(os.environ.get("MAP_REDUCE_THRESHOLD", "4000"))
MAX_REDUCE_INPUT = int(os.environ.get("MAX_REDUCE_INPUT", "20000"))

_SENTENCE_ENDS = ".!?\n"


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* characters.

    Chunk boundaries are adjusted to the nearest sentence-ending character
    (``.``, ``!``, ``?``, newline) so that sentences are not split mid-way.
    Adjacent chunks overlap by *overlap* characters for context continuity.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):  # pragma: no branch — loop always exits via break
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to break at the last sentence-ending character within the window.
        best = -1
        for i in range(end, start, -1):
            if text[i - 1] in _SENTENCE_ENDS:
                best = i
                break
        if best > start:
            end = best
        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)
    return chunks


MAX_REDUCE_DEPTH = 3  # safety cap to avoid runaway recursion


def map_reduce_summarize(
    text: str,
    summarize_fn: Callable[[str], str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    threshold: int = MAP_REDUCE_THRESHOLD,
    max_reduce_input: int = MAX_REDUCE_INPUT,
    _depth: int = 0,
) -> str:
    """Summarize *text* using map-reduce when it exceeds *threshold* length.

    Short texts are summarized in a single call.  Longer texts are chunked,
    each chunk is summarized independently (map), and then the chunk
    summaries are summarized together (reduce).

    If the combined map summaries exceed *max_reduce_input*, the reduce step
    is applied recursively (up to ``MAX_REDUCE_DEPTH`` levels) until the
    input fits.
    """
    if len(text) <= threshold:
        return summarize_fn(text)

    chunks = chunk_text(text, chunk_size, overlap)
    chunk_summaries = [summarize_fn(chunk) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    # Recursive reduce: if combined summaries are still too long, re-chunk.
    if len(combined) > max_reduce_input and _depth < MAX_REDUCE_DEPTH:
        return map_reduce_summarize(
            combined,
            summarize_fn,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
            max_reduce_input=max_reduce_input,
            _depth=_depth + 1,
        )

    return summarize_fn(combined)
