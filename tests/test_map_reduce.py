"""Tests for map_reduce.py."""

from __future__ import annotations

from dataclasses import dataclass, field

from map_reduce import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MAP_REDUCE_THRESHOLD,
    chunk_text,
    map_reduce_summarize,
)

# ---------- test doubles ----------


@dataclass
class SpySummarize:
    """Records calls and returns a canned summary."""

    result: str = "краткое содержание"
    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        return self.result


# ---------- chunk_text ----------


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        assert chunk_text("короткий текст", chunk_size=100) == ["короткий текст"]

    def test_exact_chunk_size_returns_single_chunk(self):
        text = "a" * 100
        assert chunk_text(text, chunk_size=100) == [text]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "Предложение. " * 100  # ~1400 chars
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        # All original text is covered
        joined = " ".join(chunks)
        assert "Предложение." in joined

    def test_breaks_at_sentence_boundary(self):
        text = "Первое предложение. Второе предложение. Третье."
        chunks = chunk_text(text, chunk_size=25, overlap=5)
        # First chunk should end at a sentence boundary
        assert chunks[0].endswith(".")

    def test_overlap_between_chunks(self):
        text = "Один. Два. Три. Четыре. Пять. Шесть. Семь. Восемь."
        chunks = chunk_text(text, chunk_size=20, overlap=10)
        assert len(chunks) >= 2
        # Chunks should have overlapping content
        for i in range(len(chunks) - 1):
            tail = chunks[i][-10:]
            assert tail in chunks[i + 1] or chunks[i + 1].startswith(tail[:5])

    def test_empty_string_returns_single_chunk(self):
        assert chunk_text("") == [""]

    def test_defaults_match_module_constants(self):
        # Ensure chunk_text uses module-level defaults
        text = "a" * (DEFAULT_CHUNK_SIZE + 1)
        chunks = chunk_text(text)
        assert len(chunks) > 1


# ---------- map_reduce_summarize ----------


class TestMapReduceSummarize:
    def test_short_text_single_call(self):
        spy = SpySummarize()
        result = map_reduce_summarize("короткий", spy, threshold=100)
        assert result == "краткое содержание"
        assert len(spy.calls) == 1
        assert spy.calls[0] == "короткий"

    def test_text_at_threshold_single_call(self):
        spy = SpySummarize()
        text = "x" * 100
        map_reduce_summarize(text, spy, threshold=100)
        assert len(spy.calls) == 1

    def test_long_text_map_and_reduce(self):
        spy = SpySummarize(result="сводка")
        # Create text with sentence boundaries that exceeds threshold
        text = "Предложение. " * 50  # ~700 chars
        map_reduce_summarize(text, spy, chunk_size=200, overlap=20, threshold=300)
        # map calls (one per chunk) + 1 reduce call
        assert len(spy.calls) >= 3  # at least 2 chunks + 1 reduce

    def test_reduce_receives_joined_summaries(self):
        counter = {"n": 0}

        def counting_fn(text: str) -> str:
            counter["n"] += 1
            return f"summary-{counter['n']}"

        text = "Слово. " * 200  # ~1400 chars
        result = counting_fn.__name__  # just to use the var
        result = map_reduce_summarize(
            text, counting_fn, chunk_size=200, overlap=20, threshold=300
        )
        # The last call is the reduce call; its result is what we get back
        n = counter["n"]
        assert result == f"summary-{n}"

    def test_reduce_call_contains_map_results(self):
        calls: list[str] = []

        def tracking_fn(text: str) -> str:
            calls.append(text)
            return f"[sum of: {text[:20]}]"

        text = "Предложение. " * 50
        map_reduce_summarize(
            text,
            tracking_fn,
            chunk_size=200,
            overlap=20,
            threshold=300,
        )
        # The last call (reduce) should contain map summaries
        reduce_input = calls[-1]
        assert "[sum of:" in reduce_input

    def test_defaults_match_module_constants(self):
        spy = SpySummarize()
        text = "x" * (MAP_REDUCE_THRESHOLD + 1)
        map_reduce_summarize(text, spy)
        # Should have used map-reduce (more than 1 call)
        assert len(spy.calls) > 1
