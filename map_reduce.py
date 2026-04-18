"""Map-reduce summarization for long texts.

The module is structured for testability:

* ``chunk_text`` is a pure function — easy to unit-test with plain strings.
* ``map_reduce_summarize`` takes an injectable ``summarize_fn`` so tests can
  supply a spy instead of calling a real LLM.
"""

from __future__ import annotations

from typing import Callable

DEFAULT_CHUNK_SIZE = 3000  # characters per chunk
DEFAULT_OVERLAP = 200  # character overlap between adjacent chunks
MAP_REDUCE_THRESHOLD = 4000  # texts longer than this trigger map-reduce

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
    while start < len(text):
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


def map_reduce_summarize(
    text: str,
    summarize_fn: Callable[[str], str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    threshold: int = MAP_REDUCE_THRESHOLD,
) -> str:
    """Summarize *text* using map-reduce when it exceeds *threshold* length.

    Short texts are summarized in a single call.  Longer texts are chunked,
    each chunk is summarized independently (map), and then the chunk
    summaries are summarized together (reduce).
    """
    if len(text) <= threshold:
        return summarize_fn(text)

    chunks = chunk_text(text, chunk_size, overlap)
    chunk_summaries = [summarize_fn(chunk) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)
    return summarize_fn(combined)
