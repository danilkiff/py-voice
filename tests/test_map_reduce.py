"""Tests for map_reduce.py."""

from __future__ import annotations

from dataclasses import dataclass, field

from map_reduce import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MAP_REDUCE_THRESHOLD,
    MAX_REDUCE_DEPTH,
    MAX_REDUCE_INPUT,
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

    def test_no_sentence_boundary_cuts_at_chunk_size(self):
        """When text has no sentence-ending chars, chunk_text hard-cuts."""
        text = "a" * 500  # no . ! ? \n
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) >= 2
        # First chunk is exactly chunk_size since no sentence boundary found
        assert len(chunks[0]) == 200

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


# ---------- recursive reduce ----------


class TestRecursiveReduce:
    def test_recursive_reduce_when_summaries_exceed_max(self):
        """When map summaries combined > max_reduce_input, reduce recurses."""
        calls: list[str] = []

        def shrinking_fn(text: str) -> str:
            calls.append(text)
            # Return a summary that is shorter than input but still sizable
            # enough that the first reduce pass exceeds max_reduce_input.
            return text[:80] + "..."

        # ~2100 chars, chunks of ~200 → ~10 chunks
        # Each summary ~83 chars → combined ~830+separators ≈ ~850
        # Set max_reduce_input=400 to force at least one recursion level,
        # but summaries shrink so it converges.
        text = "Предложение. " * 150
        map_reduce_summarize(
            text,
            shrinking_fn,
            chunk_size=200,
            overlap=20,
            threshold=300,
            max_reduce_input=400,
        )
        map_chunks = chunk_text(text, chunk_size=200, overlap=20)
        # More calls than a single map+reduce pass (recursion happened)
        assert len(calls) > len(map_chunks) + 1

    def test_single_reduce_when_summaries_fit(self):
        """When map summaries combined <= max_reduce_input, no recursion."""
        spy = SpySummarize(result="short")
        text = "Предложение. " * 50  # ~700 chars
        map_reduce_summarize(
            text,
            spy,
            chunk_size=200,
            overlap=20,
            threshold=300,
            max_reduce_input=99999,
        )
        map_chunks = chunk_text(text, chunk_size=200, overlap=20)
        # Exactly N map calls + 1 reduce
        assert len(spy.calls) == len(map_chunks) + 1

    def test_depth_is_capped(self):
        """Even if summaries never shrink, recursion stops at MAX_REDUCE_DEPTH."""
        calls: list[str] = []

        def inflating_fn(text: str) -> str:
            calls.append(text)
            # Always return something big — would infinite-loop without cap
            return "x" * 5000

        text = "Предложение. " * 150
        # This would recurse forever without the depth cap
        map_reduce_summarize(
            text,
            inflating_fn,
            chunk_size=200,
            overlap=20,
            threshold=300,
            max_reduce_input=100,
        )
        # Should terminate; just verify it didn't hang
        assert len(calls) > 0
        assert MAX_REDUCE_DEPTH == 3


# ---------- env var configuration ----------


class TestEnvVarDefaults:
    def test_default_chunk_size(self):
        assert DEFAULT_CHUNK_SIZE == 3000

    def test_default_overlap(self):
        assert DEFAULT_OVERLAP == 200

    def test_default_threshold(self):
        assert MAP_REDUCE_THRESHOLD == 4000

    def test_default_max_reduce_input(self):
        assert MAX_REDUCE_INPUT == 20000

    def test_env_overrides_are_documented(self):
        """The env var names are used at module level — just verify the
        constants are int (parsed from os.environ.get)."""
        assert isinstance(DEFAULT_CHUNK_SIZE, int)
        assert isinstance(MAP_REDUCE_THRESHOLD, int)
        assert isinstance(MAX_REDUCE_INPUT, int)
