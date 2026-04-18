---
status: accepted
date: 2025-05-01
---

# Character-Based Map-Reduce with Recursive Reduce

## Context and Problem Statement

Long YouTube videos produce transcripts of 100 K+ characters that exceed the context window of local LLMs (typically 8 K tokens ~ 24 K Cyrillic characters). The summarization pipeline must handle arbitrarily long texts while staying within model limits at every step, including the reduce phase where multiple chunk summaries are combined.

## Decision Drivers

* Must work within 8 K token context windows (local models like qwen3:8b)
* No external tokenizer dependency
* Deterministic, language-agnostic splitting
* Graceful handling of edge cases (very long videos, verbose model output)

## Considered Options

* **Character-based chunking with sentence-boundary alignment and recursive reduce**
* Token-based chunking (requires tokenizer library, model-specific)
* Sentence-based chunking (unpredictable chunk sizes)
* Single-pass summarization with truncation (lossy)

## Decision Outcome

Chosen option: **character-based map-reduce with recursive reduce**.

**Chunking** — `chunk_text()` splits text at approximately `CHUNK_SIZE` (default 3000) characters, backtracking to the nearest sentence-ending character (`.!?\n`) to avoid mid-sentence breaks. Adjacent chunks overlap by `OVERLAP` (default 200) characters for context continuity.

**Map-reduce** — `map_reduce_summarize()` applies the summarize function to each chunk independently (map), joins the resulting summaries, and summarizes the combined text (reduce). If the combined summaries exceed `MAX_REDUCE_INPUT` (default 20 000 characters), the reduce step recurses: the combined text is re-chunked and map-reduced again. Recursion is capped at `MAX_REDUCE_DEPTH = 3` to prevent runaway loops if the model produces verbose output.

**Configuration** — `CHUNK_SIZE`, `MAP_REDUCE_THRESHOLD`, and `MAX_REDUCE_INPUT` are configurable via environment variables, allowing users with larger-context models (32 K+) to increase limits and reduce the number of LLM calls.

Character counting was chosen over token counting because it requires no tokenizer dependency, works identically for any language, and the conservative default (3000 chars ~ 750-900 tokens) leaves ample room for the prompt template within an 8 K token window.

### Consequences

* Good, because no tokenizer dependency is needed
* Good, because recursive reduce handles arbitrarily long videos without silent truncation
* Good, because depth cap prevents infinite recursion if the model is verbose
* Good, because env var overrides allow tuning for larger-context models
* Bad, because character count is an imprecise proxy for token count (may under-utilize context)
* Bad, because multiple recursion levels multiply LLM calls and add latency (e.g., 92 calls for a 3-hour video)
* Bad, because each summarization pass loses some information (compounding over recursion levels)
