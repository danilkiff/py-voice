# PY Voice

Web app for Russian audio transcription (mp3, wav, m4a, flac, ogg).

## Dependencies

- required: Python 3.11+, uv, ffmpeg
- optional: NVIDIA driver + CUDA 12.x runtime

## Cheat sheet

```bash
# setup: CPU-only
uv sync
# setup: Linux + CUDA
uv sync --extra cuda
# run tests
uv run pytest
# preload: Systran/faster-whisper-{tiny,base,small,medium,large-v3}
# ~/.cache/huggingface/hub/ or override HF_HOME
uv run hf download Systran/faster-whisper-large-v3
# run python stack: UI 0.0.0.0:7860
uv run main.py
# run compose stack: UI on 0.0.0.0:7860, Ollama on 0.0.0.0:11434
docker compose up --build
```

## License

[Unlicense](LICENSE) — public domain.
