"""Gradio web app for Russian audio transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

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

TranscribeFn = Callable[[str, str], TranscriptionResult]
SummarizeFn = Callable[[str], str]


def _default_transcribe(audio_path: str, model_size: str) -> TranscriptionResult:
    return get_transcriber(model_size).transcribe(audio_path)


def _default_summarize(text: str) -> str:
    cfg = load_config()
    return Summarizer(cfg.base_url, cfg.model).summarize(text)


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
) -> Callable[[str | None, str], tuple[str, str | None]]:
    """Build a Gradio click handler. transcribe_fn is injectable for tests.

    The default is resolved at call time (not at def-time) so monkeypatching
    `_default_transcribe` in tests has the expected effect.
    """
    fn: TranscribeFn = (
        transcribe_fn if transcribe_fn is not None else _default_transcribe
    )

    def run(audio_path: str | None, model_size: str) -> tuple[str, str | None]:
        if not audio_path:
            return EMPTY_INPUT_MESSAGE, None

        result = fn(audio_path, model_size)
        text = format_header(result, model_size) + result.text
        out_path = write_text_file(Path(audio_path).stem, text, output_dir)
        return text, str(out_path)

    return run


def make_summarize_handler(
    summarize_fn: SummarizeFn | None = None,
) -> Callable[[str], str]:
    """Build a Gradio click handler for summarization. summarize_fn is injectable for tests."""
    fn: SummarizeFn = summarize_fn if summarize_fn is not None else _default_summarize

    def summarize(transcript: str) -> str:
        if not transcript or not transcript.strip():
            return EMPTY_TRANSCRIPT_MESSAGE
        try:
            return fn(transcript)
        except Exception:
            return OLLAMA_UNAVAILABLE

    return summarize


def build_app(
    transcribe_fn: TranscribeFn | None = None,
    summarize_fn: SummarizeFn | None = None,
) -> gr.Blocks:
    run_handler = make_run_handler(transcribe_fn)
    summarize_handler = make_summarize_handler(summarize_fn)

    with gr.Blocks(title="py-voice — Russian transcription") as app:
        gr.Markdown(
            "# py-voice\n"
            "Транскрибация аудио на русском языке через **faster-whisper**.\n"
            "Поддерживаются mp3, wav, m4a, flac, ogg и другие форматы (через ffmpeg)."
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
                    info="large-v3 даёт лучшее качество. tiny/base — быстрые, для тестов.",
                )
                run_button = gr.Button("Транскрибировать", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(
                    label="Результат",
                    lines=18,
                )
                file_output = gr.File(label="Скачать .txt")

        with gr.Row():
            summarize_button = gr.Button("Суммаризировать", variant="secondary")

        with gr.Row():
            summary_output = gr.Textbox(
                label="Краткое содержание",
                lines=8,
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

    return app
