"""Benchmark the YouTube summarization pipeline end-to-end."""

from __future__ import annotations

import sys
import time

from config import load_config
from map_reduce import chunk_text, map_reduce_summarize
from summarizer import Summarizer
from transcriber import get_transcriber
from youtube import (
    _default_extract_info,
    download_audio,
    fetch_subtitles,
    validate_youtube_url,
)


def _debug_subtitle_info(url: str) -> None:
    """Print raw subtitle metadata from yt-dlp to diagnose missing subs."""
    opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["ru", "en"],
        "skip_download": True,
        "quiet": True,
    }
    info = _default_extract_info(url, opts)
    print(f"  title: {info.get('title', '?')}")
    print(f"  duration: {info.get('duration', '?')}s")

    for key in ("subtitles", "automatic_captions"):
        subs = info.get(key) or {}
        if subs:
            langs = list(subs.keys())
            print(f"  {key}: {len(langs)} lang(s) → {langs[:10]}")
            for lang in ("ru", "en"):
                entries = subs.get(lang, [])
                if entries:
                    exts = [e.get("ext", "?") for e in entries]
                    has_data = [bool(e.get("data")) for e in entries]
                    has_url = [bool(e.get("url")) for e in entries]
                    print(
                        f"    [{lang}] {len(entries)} entries: "
                        f"exts={exts[:5]}  has_data={has_data[:5]}  has_url={has_url[:5]}"
                    )
        else:
            print(f"  {key}: (empty)")


def main(url: str) -> None:
    validate_youtube_url(url)
    cfg = load_config()
    print(f"[config]     ollama={cfg.base_url}  model={cfg.model}")
    summarizer = Summarizer(cfg.base_url, cfg.model)

    # --- Step 0: debug subtitle info ---
    print("\n[debug] fetching video metadata...")
    t0 = time.perf_counter()
    _debug_subtitle_info(url)
    t_meta = time.perf_counter() - t0
    print(f"[debug]      metadata time={t_meta:.2f}s")

    # --- Step 1: try subtitles ---
    print("\n[subtitles]  fetching...")
    t0 = time.perf_counter()
    sub_result = fetch_subtitles(url)
    t_subs = time.perf_counter() - t0

    if sub_result is not None:
        print(
            f"[subtitles]  ✓ lang={sub_result.language}  "
            f"len={len(sub_result.text)} chars  time={t_subs:.2f}s"
        )
        print(f"[subtitles]  preview: {sub_result.text[:200]}...")
        text = sub_result.text
    else:
        print(f"[subtitles]  ✗ not found  time={t_subs:.2f}s")

        # --- Step 2: download audio ---
        print("\n[download]   downloading audio...")
        t0 = time.perf_counter()
        audio_path = download_audio(url)
        t_dl = time.perf_counter() - t0
        print(f"[download]   ✓ path={audio_path}  time={t_dl:.2f}s")

        # --- Step 3: transcribe ---
        print("\n[transcribe] loading model and transcribing...")
        t0 = time.perf_counter()
        result = get_transcriber("large-v3").transcribe(str(audio_path))
        t_tr = time.perf_counter() - t0
        print(
            f"[transcribe] ✓ lang={result.language}  duration={result.duration:.1f}s  "
            f"len={len(result.text)} chars  time={t_tr:.2f}s"
        )
        print(f"[transcribe] preview: {result.text[:200]}...")
        text = result.text

    # --- Step 4: chunk ---
    chunks = chunk_text(text)
    print(f"\n[chunk]      {len(chunks)} chunk(s), sizes: {[len(c) for c in chunks]}")

    # --- Step 5: map-reduce summarize ---
    print("\n[summarize]  starting map-reduce...")
    call_n = 0
    total_chunks = len(chunks)
    original_summarize = summarizer.summarize
    SEP = "-" * 60

    def verbose_summarize(t: str) -> str:
        nonlocal call_n
        call_n += 1
        is_reduce = call_n > total_chunks
        phase = "REDUCE" if is_reduce else f"map[{call_n}/{total_chunks}]"

        print(f"\n  {SEP}")
        print(f"  [{phase}] input: {len(t)} chars")
        # Show input: first/last 300 chars
        if len(t) <= 620:
            print(f"  IN:  {t}")
        else:
            print(f"  IN:  {t[:300]}")
            print(f"  ...({len(t) - 600} chars skipped)...")
            print(f"  ...  {t[-300:]}")

        t0 = time.perf_counter()
        result = original_summarize(t)
        elapsed = time.perf_counter() - t0

        print(f"  OUT ({len(result)} chars, {elapsed:.2f}s):")
        print(f"  {result}")
        return result

    t0 = time.perf_counter()
    summary = map_reduce_summarize(text, verbose_summarize)
    t_sum = time.perf_counter() - t0
    print(f"\n[summarize]  ✓ total={len(summary)} chars  time={t_sum:.2f}s")

    print(f"\n{'=' * 60}")
    print(summary)


if __name__ == "__main__":
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://www.youtube.com/watch?v=9lO06Zxhu88"
    )
    main(url)
