---
status: accepted
date: 2025-05-01
---

# Use faster-whisper for Local Speech-to-Text

## Context and Problem Statement

py-voice needs a speech-to-text engine for Russian audio transcription. The engine must handle long audio files, support GPU acceleration when available, and work offline without per-request API costs.

## Decision Drivers

* Privacy: audio must not leave the local network
* Cost: no per-request billing for transcription
* Quality: Russian language support must be strong
* Hardware flexibility: must work on both CPU-only and NVIDIA GPU machines

## Considered Options

* **faster-whisper** (CTranslate2 backend for OpenAI Whisper)
* Cloud APIs (Google Cloud Speech-to-Text, Azure AI Speech)
* Original OpenAI Whisper (PyTorch backend)
* Other local models (Coqui STT, Vosk)

## Decision Outcome

Chosen option: **faster-whisper**, because it provides the best balance of quality, speed, and hardware flexibility for offline Russian transcription.

faster-whisper uses CTranslate2 as its inference backend, which enables float16 on GPU and int8 quantization on CPU without requiring PyTorch. This also allows CUDA auto-detection via `ctranslate2.get_cuda_device_count()` (in `device.py`), avoiding a heavy PyTorch dependency just for hardware probing.

Multiple model sizes are offered (tiny through large-v3) so users can trade quality for speed. The default is `large-v3` for best Russian transcription quality.

On Linux with pip-installed NVIDIA wheels, CUDA shared libraries are preloaded via `ctypes.CDLL(..., RTLD_GLOBAL)` in `transcriber.py` to work around dynamic linker issues.

### Consequences

* Good, because transcription is free, offline, and private
* Good, because CTranslate2 is already a transitive dependency, so CUDA detection adds no extra weight
* Good, because int8 CPU fallback makes the app usable without a GPU
* Bad, because the large-v3 model is ~3 GB to download
* Bad, because CPU inference is significantly slower than GPU
* Bad, because the default language is hardcoded to Russian (`"ru"`)
