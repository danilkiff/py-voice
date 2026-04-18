"""Gradio web app for Russian audio transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Iterator

import gradio as gr

from config import load_config
from summarizer import Summarizer
from transcriber import TranscriptionResult, get_transcriber

MODEL_CHOICES: list[str] = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
]
DEFAULT_MODEL = "large-v3"
EMPTY_INPUT_MESSAGE = "Загрузите аудиофайл (mp3 или wav)."
EMPTY_TRANSCRIPT_MESSAGE = "Сначала выполните транскрибацию."
OLLAMA_UNAVAILABLE = "Ollama недоступен. Убедитесь, что сервис запущен."
EMPTY_URL_MESSAGE = "Введите ссылку на YouTube видео."
YOUTUBE_ERROR = "Не удалось обработать видео. Проверьте ссылку и попробуйте снова."

TranscribeFn = Callable[[str, str], TranscriptionResult]
SummarizeFn = Callable[[str], str]
YouTubeSummarizeFn = Callable[[str], Iterator[str]]


def _default_transcribe(audio_path: str, model_size: str) -> TranscriptionResult:
    return get_transcriber(model_size).transcribe(audio_path)


def _default_summarize(text: str) -> str:
    cfg = load_config()
    return Summarizer(cfg.base_url, cfg.model).summarize(text)


def _format_timecode(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_thumbnail_html(storyboard: object, seconds: float) -> str:
    """Return an <img> tag with CSS cropping for a storyboard sprite."""
    from youtube import StoryboardInfo, thumbnail_url_for_time

    if not isinstance(storyboard, StoryboardInfo):
        return ""
    sprite_url, x, y = thumbnail_url_for_time(storyboard, seconds)
    w, h = storyboard.width, storyboard.height
    return (
        f'<img src="{sprite_url}" '
        f'style="object-fit:none;object-position:-{x}px -{y}px;'
        f'width:{w}px;height:{h}px;">'
    )


def _group_timed_segments(
    segments: tuple[tuple[float, str], ...],
    chunk_size: int,
) -> list[tuple[float, str]]:
    """Group (time, text) segments by cumulative character length."""
    if not segments:
        return []
    groups: list[tuple[float, str]] = []
    current_time = segments[0][0]
    current_parts: list[str] = []
    current_len = 0
    for start, text in segments:
        if current_len + len(text) > chunk_size and current_parts:
            groups.append((current_time, " ".join(current_parts)))
            current_time = start
            current_parts = []
            current_len = 0
        current_parts.append(text)
        current_len += len(text)
    if current_parts:  # pragma: no branch — always true when segments non-empty
        groups.append((current_time, " ".join(current_parts)))
    return groups


def _streaming_reduce(
    combined: str,
    summarize_fn: Callable[[str], str],
    output_prefix: str,
) -> Iterator[str]:
    """Reduce phase with per-call progress yields."""
    from map_reduce import MAX_REDUCE_DEPTH, MAX_REDUCE_INPUT, chunk_text

    text = combined
    depth = 0
    while len(text) > MAX_REDUCE_INPUT and depth < MAX_REDUCE_DEPTH:
        reduce_chunks = chunk_text(text)
        reduce_sums: list[str] = []
        for j, rc in enumerate(reduce_chunks):
            yield output_prefix + (
                f"---\n\n*Формирование итога… ({j + 1}/{len(reduce_chunks)})*\n"
            )
            reduce_sums.append(summarize_fn(rc))
        text = "\n\n".join(reduce_sums)
        depth += 1

    yield output_prefix + "---\n\n*Финальная суммаризация…*\n"
    yield summarize_fn(text)


def _default_youtube_summarize(url: str) -> Iterator[str]:
    from map_reduce import (
        DEFAULT_CHUNK_SIZE,
        MAP_REDUCE_THRESHOLD,
        chunk_text,
    )
    from youtube import download_audio, fetch_video_info, validate_youtube_url

    validate_youtube_url(url)
    cfg = load_config()
    summarizer = Summarizer(cfg.base_url, cfg.model)

    yield "Получение субтитров…"
    video_info = fetch_video_info(url)

    if video_info.subtitles is not None:
        text = video_info.subtitles.text
        segments = video_info.subtitles.segments
        storyboard = video_info.storyboard
    else:
        # Slow path: download audio, transcribe
        yield "Субтитры не найдены. Загрузка аудио…"
        audio_path = download_audio(url)
        yield "Транскрибация аудио…"
        transcription = get_transcriber().transcribe(str(audio_path))
        text = transcription.text
        segments = ()
        storyboard = video_info.storyboard

    # Short text — single call, no streaming needed
    if len(text) <= MAP_REDUCE_THRESHOLD:
        yield "Суммаризация…"
        yield summarizer.summarize(text)
        return

    # Timed path: chunk by segments with timecodes + thumbnails
    if segments:
        timed_chunks = _group_timed_segments(segments, DEFAULT_CHUNK_SIZE)
        output = ""
        summaries: list[str] = []
        for i, (start_time, chunk) in enumerate(timed_chunks):
            summary = summarizer.summarize(chunk)
            summaries.append(summary)
            timecode = _format_timecode(start_time)
            thumb = _build_thumbnail_html(storyboard, start_time) if storyboard else ""
            output += (
                f"**[{timecode}]** {summary}\n\n{thumb}\n\n"
                if thumb
                else f"**[{timecode}]** {summary}\n\n"
            )
            yield output

        combined = "\n\n".join(summaries)
        yield from _streaming_reduce(combined, summarizer.summarize, output)
        return

    # Untimed fallback: use chunk_text as before
    chunks = chunk_text(text)
    output = ""
    summaries_plain: list[str] = []
    for i, chunk in enumerate(chunks):
        summary = summarizer.summarize(chunk)
        summaries_plain.append(summary)
        output += f"**[{i + 1}/{len(chunks)}]** {summary}\n\n"
        yield output

    combined = "\n\n".join(summaries_plain)
    yield from _streaming_reduce(combined, summarizer.summarize, output)


def format_header(result: TranscriptionResult, model_size: str) -> str:
    return (
        f"Язык: {result.language} | Длительность: {result.duration:.1f} с | "
        f"Модель: {model_size}\n\n"
    )


def write_text_file(stem: str, text: str, directory: Path | None = None) -> Path:
    target_dir = directory if directory is not None else Path(tempfile.gettempdir())
    out_path = target_dir / f"{stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def make_run_handler(
    transcribe_fn: TranscribeFn | None = None,
    output_dir: Path | None = None,
) -> Callable[[str | None, str], Iterator[tuple[str, str | None]]]:
    """Build a Gradio click handler. transcribe_fn is injectable for tests.

    The default is resolved at call time (not at def-time) so monkeypatching
    `_default_transcribe` in tests has the expected effect.
    """
    fn: TranscribeFn = (
        transcribe_fn if transcribe_fn is not None else _default_transcribe
    )

    def run(
        audio_path: str | None, model_size: str
    ) -> Iterator[tuple[str, str | None]]:
        if not audio_path:
            yield EMPTY_INPUT_MESSAGE, None
            return

        yield "Транскрибация…", None
        result = fn(audio_path, model_size)
        text = format_header(result, model_size) + result.text
        out_path = write_text_file(Path(audio_path).stem, text, output_dir)
        yield text, str(out_path)

    return run


def make_summarize_handler(
    summarize_fn: SummarizeFn | None = None,
) -> Callable[[str], Iterator[str]]:
    """Build a Gradio click handler for summarization. summarize_fn is injectable for tests."""
    fn: SummarizeFn = summarize_fn if summarize_fn is not None else _default_summarize

    def summarize(transcript: str) -> Iterator[str]:
        if not transcript or not transcript.strip():
            yield EMPTY_TRANSCRIPT_MESSAGE
            return
        try:
            yield "Суммаризация…"
            yield fn(transcript)
        except Exception:
            yield OLLAMA_UNAVAILABLE

    return summarize


def make_youtube_handler(
    youtube_summarize_fn: YouTubeSummarizeFn | None = None,
) -> Callable[[str], Iterator[str]]:
    """Build a Gradio streaming handler for YouTube summarization.

    The returned function is a generator — Gradio updates the output
    textbox after each ``yield``, giving the user progressive feedback
    during the map phase.
    """
    fn: YouTubeSummarizeFn = (
        youtube_summarize_fn
        if youtube_summarize_fn is not None
        else _default_youtube_summarize
    )

    def handle_youtube(url: str) -> Iterator[str]:
        if not url or not url.strip():
            yield EMPTY_URL_MESSAGE
            return
        try:
            yield from fn(url.strip())
        except ValueError as exc:
            yield str(exc)
        except Exception:
            yield YOUTUBE_ERROR

    return handle_youtube


def build_app(
    transcribe_fn: TranscribeFn | None = None,
    summarize_fn: SummarizeFn | None = None,
    youtube_summarize_fn: YouTubeSummarizeFn | None = None,
) -> gr.Blocks:
    run_handler = make_run_handler(transcribe_fn)
    summarize_handler = make_summarize_handler(summarize_fn)
    youtube_handler = make_youtube_handler(youtube_summarize_fn)

    with gr.Blocks(title="py-voice — Russian transcription") as app:
        gr.Markdown("# py-voice")

        with gr.Tabs():
            with gr.Tab("Аудио"):
                gr.Markdown(
                    "Транскрибация аудио на русском языке через "
                    "**faster-whisper**.\n"
                    "Поддерживаются mp3, wav, m4a, flac, ogg и другие "
                    "форматы (через ffmpeg)."
                )

                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Аудиофайл",
                            type="filepath",
                            sources=["upload"],
                        )
                        model_dropdown = gr.Dropdown(
                            label="Модель",
                            choices=MODEL_CHOICES,
                            value=DEFAULT_MODEL,
                            info="large-v3 даёт лучшее качество. "
                            "tiny/base — быстрые, для тестов.",
                        )
                        run_button = gr.Button(
                            "Транскрибировать",
                            variant="primary",
                        )

                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Результат",
                            lines=18,
                        )
                        file_output = gr.File(label="Скачать .txt")

                with gr.Row():
                    summarize_button = gr.Button(
                        "Суммаризировать",
                        variant="secondary",
                    )

                with gr.Row():
                    summary_output = gr.Markdown(
                        label="Краткое содержание",
                    )

                run_button.click(
                    fn=run_handler,
                    inputs=[audio_input, model_dropdown],
                    outputs=[text_output, file_output],
                )
                summarize_button.click(
                    fn=summarize_handler,
                    inputs=[text_output],
                    outputs=[summary_output],
                )

            with gr.Tab("YouTube"):
                gr.Markdown(
                    "Суммаризация YouTube-видео: субтитры (если есть) или "
                    "транскрипция аудио → краткое содержание через LLM."
                )

                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(
                            label="Ссылка на YouTube",
                            placeholder="https://www.youtube.com/watch?v=...",
                        )
                        yt_button = gr.Button(
                            "Суммаризировать видео",
                            variant="primary",
                        )

                    with gr.Column():
                        yt_output = gr.Markdown(
                            label="Краткое содержание",
                        )

                yt_button.click(
                    fn=youtube_handler,
                    inputs=[url_input],
                    outputs=[yt_output],
                )

    return app
