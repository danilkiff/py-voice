"""Russian speech-to-text via faster-whisper.

The module is structured for testability:

* `validate_audio_path` and `join_segments` are pure functions.
* `Transcriber` takes an injectable `loader`, so tests can supply fakes
  instead of loading the real ~3 GB model.
* `get_transcriber` is a tiny registry that caches one Transcriber per
  model size — and can be cleared with `clear_transcriber_cache`.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from device import pick_device_and_compute_type


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    duration: float


class _Segment(Protocol):
    text: str


class _Info(Protocol):
    language: str
    duration: float


class _Model(Protocol):
    def transcribe(
        self, audio: str, **kwargs: Any
    ) -> tuple[Iterable[_Segment], _Info]: ...


ModelLoader = Callable[[str, str, str], _Model]
CDLLLoader = Callable[[str], Any]


def _default_cdll_loader(path: str) -> Any:
    """Default ctypes loader: dlopen with RTLD_GLOBAL so symbols are visible."""
    import ctypes

    return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _discover_nvidia_lib_dirs() -> list[Path]:
    """Find nvidia/cublas/lib and nvidia/cudnn/lib directories from pip wheels.

    `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` ship as PEP 420 namespace
    packages without `__init__.py`, so the conventional `__file__` trick
    used in faster-whisper docs returns None. We walk the package's
    `__path__` instead.
    """
    found: list[Path] = []
    for pkg_name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            pkg = __import__(pkg_name, fromlist=[""])
        except ImportError:
            continue
        for entry in getattr(pkg, "__path__", []):
            lib_dir = Path(entry) / "lib"
            if lib_dir.is_dir():
                found.append(lib_dir)
                break
    return found


def _preload_cuda_libs(
    *,
    system: str | None = None,
    lib_dirs: Iterable[Path] | None = None,
    cdll: CDLLLoader | None = None,
) -> int:
    """Preload cuBLAS / cuDNN .so files into the global symbol table.

    On Linux, ctranslate2 fails with `Library libcublas.so.12 is not found
    or cannot be loaded` when those libs come from pip-installed `nvidia-*`
    wheels — the wheels do not register their paths with the dynamic
    loader, so `LD_LIBRARY_PATH` would normally be required. Loading each
    .so via `ctypes.CDLL(..., RTLD_GLOBAL)` makes the SONAMEs visible in
    the process so subsequent dlopen calls find them by name.

    No-op on macOS / Windows. Silently skips libs that fail to load.
    Returns the number of successfully loaded shared objects.
    """
    if (system or platform.system()) != "Linux":
        return 0

    dirs = list(lib_dirs) if lib_dirs is not None else _discover_nvidia_lib_dirs()
    loader = cdll if cdll is not None else _default_cdll_loader

    loaded = 0
    for lib_dir in dirs:
        if not lib_dir.is_dir():
            continue
        for so in sorted(lib_dir.glob("*.so*")):
            try:
                loader(str(so))
                loaded += 1
            except OSError:
                pass
    return loaded


def _default_model_loader(model_size: str, device: str, compute_type: str) -> _Model:
    """Lazy import so importing this module is cheap and tests can avoid it."""
    if device == "cuda":
        _preload_cuda_libs()
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def validate_audio_path(audio_path: str | Path) -> Path:
    """Return a Path, raising FileNotFoundError unless it points to a real file."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def join_segments(segments: Iterable[_Segment]) -> str:
    """Concatenate Whisper segment texts into a single cleaned line."""
    return " ".join(segment.text.strip() for segment in segments).strip()


class Transcriber:
    """Holds one faster-whisper model and runs transcription against it."""

    def __init__(
        self,
        model_size: str = "large-v3",
        *,
        device: str | None = None,
        compute_type: str | None = None,
        loader: ModelLoader | None = None,
    ) -> None:
        if device is None or compute_type is None:
            auto_device, auto_compute = pick_device_and_compute_type()
            device = device or auto_device
            compute_type = compute_type or auto_compute
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._loader: ModelLoader = loader or _default_model_loader
        self._model: _Model | None = None

    @property
    def model(self) -> _Model:
        """Lazy-load the underlying model on first access."""
        if self._model is None:
            self._model = self._loader(self.model_size, self.device, self.compute_type)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str = "ru",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        path = validate_audio_path(audio_path)
        segments, info = self.model.transcribe(
            str(path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        return TranscriptionResult(
            text=join_segments(segments),
            language=info.language,
            duration=info.duration,
        )


_TRANSCRIBERS: dict[str, Transcriber] = {}


def get_transcriber(model_size: str = "large-v3") -> Transcriber:
    """Return a cached Transcriber for the given model size."""
    if model_size not in _TRANSCRIBERS:
        _TRANSCRIBERS[model_size] = Transcriber(model_size)
    return _TRANSCRIBERS[model_size]


def clear_transcriber_cache() -> None:
    """Drop all cached Transcriber instances. Used by tests."""
    _TRANSCRIBERS.clear()
