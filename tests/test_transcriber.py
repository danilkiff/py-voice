"""Tests for transcriber.py."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.conftest import FakeInfo, FakeModel, FakeSegment, RecordingLoader
from transcriber import (
    Transcriber,
    TranscriptionResult,
    _discover_nvidia_lib_dirs,
    _preload_cuda_libs,
    clear_transcriber_cache,
    get_transcriber,
    join_segments,
    validate_audio_path,
)

# ---------- TranscriptionResult ----------


class TestTranscriptionResult:
    def test_holds_fields(self):
        result = TranscriptionResult(text="hi", language="ru", duration=2.0)
        assert result.text == "hi"
        assert result.language == "ru"
        assert result.duration == 2.0

    def test_is_frozen(self):
        result = TranscriptionResult(text="hi", language="ru", duration=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.text = "bye"  # type: ignore[misc]


# ---------- validate_audio_path ----------


class TestValidateAudioPath:
    def test_existing_file_returns_path(self, audio_file):
        result = validate_audio_path(audio_file)
        assert result == audio_file
        assert isinstance(result, Path)

    def test_accepts_string_path(self, audio_file):
        result = validate_audio_path(str(audio_file))
        assert result == audio_file

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nope.wav"
        with pytest.raises(FileNotFoundError, match="nope.wav"):
            validate_audio_path(missing)

    def test_directory_is_not_a_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_audio_path(tmp_path)


# ---------- join_segments ----------


class TestJoinSegments:
    def test_empty_iterable(self):
        assert join_segments([]) == ""

    def test_single_segment_strips_whitespace(self):
        assert join_segments([FakeSegment(text="  привет  ")]) == "привет"

    def test_multiple_segments_joined_with_single_space(self):
        segs = [
            FakeSegment(text="один"),
            FakeSegment(text="два"),
            FakeSegment(text="три"),
        ]
        assert join_segments(segs) == "один два три"

    def test_each_segment_individually_stripped(self):
        segs = [FakeSegment(text=" а "), FakeSegment(text=" б ")]
        assert join_segments(segs) == "а б"

    def test_blank_segments_collapse_to_empty(self):
        segs = [FakeSegment(text="   "), FakeSegment(text=" ")]
        assert join_segments(segs) == ""

    def test_works_with_generator(self):
        gen = (FakeSegment(text=t) for t in ["a", "b"])
        assert join_segments(gen) == "a b"


# ---------- Transcriber init ----------


class TestTranscriberInit:
    def test_auto_device_when_not_specified(self, loader):
        t = Transcriber(model_size="tiny", loader=loader)
        assert t.device in {"cuda", "cpu"}
        assert t.compute_type in {"float16", "int8"}
        assert t.model_size == "tiny"

    def test_explicit_device_and_compute_type(self, loader):
        t = Transcriber(
            model_size="base",
            device="cuda",
            compute_type="float16",
            loader=loader,
        )
        assert t.device == "cuda"
        assert t.compute_type == "float16"

    def test_explicit_device_only_keeps_partial_override(self, loader):
        t = Transcriber(model_size="tiny", device="cuda", loader=loader)
        assert t.device == "cuda"
        # compute_type was filled in by the auto-picker
        assert t.compute_type in {"float16", "int8"}

    def test_default_loader_used_when_none_provided(self):
        t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
        from transcriber import _default_model_loader

        assert t._loader is _default_model_loader


# ---------- Transcriber.model lazy loading ----------


class TestTranscriberModelProperty:
    def test_loader_not_called_on_init(self, loader):
        Transcriber(model_size="tiny", device="cpu", compute_type="int8", loader=loader)
        assert loader.calls == []

    def test_loader_called_on_first_access(self, loader, fake_model):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        assert t.model is fake_model
        assert loader.calls == [("tiny", "cpu", "int8")]

    def test_loader_cached_after_first_access(self, loader, fake_model):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        first = t.model
        second = t.model
        assert first is second
        assert len(loader.calls) == 1


# ---------- Transcriber.transcribe ----------


class TestTranscriberTranscribe:
    def test_returns_transcription_result(self, loader, audio_file):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        result = t.transcribe(audio_file)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "привет"
        assert result.language == "ru"
        assert result.duration == 1.5

    def test_passes_default_kwargs_to_model(self, loader, audio_file, fake_model):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        t.transcribe(audio_file)
        audio_arg, kwargs = fake_model.calls[0]
        assert audio_arg == str(audio_file)
        assert kwargs == {"language": "ru", "beam_size": 5, "vad_filter": True}

    def test_passes_overridden_kwargs(self, loader, audio_file, fake_model):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        t.transcribe(audio_file, language="en", beam_size=1, vad_filter=False)
        _, kwargs = fake_model.calls[0]
        assert kwargs == {"language": "en", "beam_size": 1, "vad_filter": False}

    def test_missing_file_raises_before_model_load(self, loader, tmp_path):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        with pytest.raises(FileNotFoundError):
            t.transcribe(tmp_path / "ghost.wav")
        # Model should not have been loaded since validation happens first.
        assert loader.calls == []

    def test_accepts_string_path(self, loader, audio_file):
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        result = t.transcribe(str(audio_file))
        assert result.text == "привет"

    def test_uses_custom_segments_and_info(self, audio_file):
        custom_model = FakeModel(
            segments=[FakeSegment(text="один"), FakeSegment(text="два")],
            info=FakeInfo(language="ru", duration=12.34),
        )
        loader = RecordingLoader(model=custom_model)
        t = Transcriber(
            model_size="tiny", device="cpu", compute_type="int8", loader=loader
        )
        result = t.transcribe(audio_file)
        assert result.text == "один два"
        assert result.duration == 12.34


# ---------- registry: get_transcriber / clear_transcriber_cache ----------


class TestTranscriberRegistry:
    def test_get_transcriber_returns_same_instance_for_same_size(
        self, monkeypatch, loader
    ):
        # Patch loader so no real model is downloaded.
        monkeypatch.setattr("transcriber._default_model_loader", loader)
        a = get_transcriber("tiny")
        b = get_transcriber("tiny")
        assert a is b

    def test_get_transcriber_different_sizes_are_distinct(self, monkeypatch, loader):
        monkeypatch.setattr("transcriber._default_model_loader", loader)
        a = get_transcriber("tiny")
        b = get_transcriber("base")
        assert a is not b
        assert a.model_size == "tiny"
        assert b.model_size == "base"

    def test_clear_cache_drops_instances(self, monkeypatch, loader):
        monkeypatch.setattr("transcriber._default_model_loader", loader)
        a = get_transcriber("tiny")
        clear_transcriber_cache()
        b = get_transcriber("tiny")
        assert a is not b


# ---------- default loader smoke test ----------


class TestDefaultModelLoader:
    def test_calls_whisper_model_with_args(self, monkeypatch):
        """The default loader should construct WhisperModel(model_size, device=, compute_type=)."""
        from transcriber import _default_model_loader

        captured = {}

        class FakeWhisperModel:
            def __init__(self, model_size, device, compute_type):
                captured["args"] = (model_size, device, compute_type)

        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)

        instance = _default_model_loader("tiny", "cpu", "int8")
        assert isinstance(instance, FakeWhisperModel)
        assert captured["args"] == ("tiny", "cpu", "int8")

    def test_cuda_path_invokes_preload(self, monkeypatch):
        """When device='cuda', the loader should preload CUDA libs first."""
        import transcriber

        called = []
        monkeypatch.setattr(
            transcriber,
            "_preload_cuda_libs",
            lambda: called.append("preload") or 0,
        )

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                called.append("whisper")

        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
        transcriber._default_model_loader("tiny", "cuda", "float16")
        assert called == ["preload", "whisper"]

    def test_cpu_path_skips_preload(self, monkeypatch):
        """When device='cpu', the loader should NOT call the CUDA preload."""
        import transcriber

        def boom() -> int:
            raise AssertionError("preload must not be called for cpu")

        monkeypatch.setattr(transcriber, "_preload_cuda_libs", boom)

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                pass

        import faster_whisper

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
        transcriber._default_model_loader("tiny", "cpu", "int8")


# ---------- _preload_cuda_libs ----------


class TestPreloadCudaLibs:
    def test_noop_on_macos(self):
        assert _preload_cuda_libs(system="Darwin") == 0

    def test_noop_on_windows(self):
        assert _preload_cuda_libs(system="Windows") == 0

    def test_noop_on_linux_when_no_lib_dirs(self):
        assert _preload_cuda_libs(system="Linux", lib_dirs=[]) == 0

    def test_skips_nonexistent_dirs(self, tmp_path):
        ghost = tmp_path / "ghost"
        loaded = _preload_cuda_libs(
            system="Linux",
            lib_dirs=[ghost],
            cdll=lambda _path: None,
        )
        assert loaded == 0

    def test_loads_each_so_in_dir(self, tmp_path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "libcublas.so.12").write_bytes(b"")
        (lib_dir / "libcudnn.so.9").write_bytes(b"")
        (lib_dir / "libcudnn_ops.so.9").write_bytes(b"")
        (lib_dir / "README.txt").write_text("not a shared object")

        seen: list[str] = []

        def fake_cdll(path: str) -> None:
            seen.append(Path(path).name)

        loaded = _preload_cuda_libs(
            system="Linux",
            lib_dirs=[lib_dir],
            cdll=fake_cdll,
        )
        assert loaded == 3
        assert sorted(seen) == sorted(
            ["libcublas.so.12", "libcudnn.so.9", "libcudnn_ops.so.9"]
        )
        assert "README.txt" not in seen

    def test_oserror_is_silently_skipped(self, tmp_path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "libgood.so.1").write_bytes(b"")
        (lib_dir / "libbad.so.1").write_bytes(b"")

        def selective_cdll(path: str) -> None:
            if "bad" in path:
                raise OSError("simulated dlopen failure")

        loaded = _preload_cuda_libs(
            system="Linux",
            lib_dirs=[lib_dir],
            cdll=selective_cdll,
        )
        assert loaded == 1

    def test_iterates_multiple_lib_dirs(self, tmp_path):
        cublas_dir = tmp_path / "cublas"
        cudnn_dir = tmp_path / "cudnn"
        cublas_dir.mkdir()
        cudnn_dir.mkdir()
        (cublas_dir / "libcublas.so.12").write_bytes(b"")
        (cudnn_dir / "libcudnn.so.9").write_bytes(b"")

        loaded_paths: list[str] = []
        loaded = _preload_cuda_libs(
            system="Linux",
            lib_dirs=[cublas_dir, cudnn_dir],
            cdll=lambda p: loaded_paths.append(p),
        )
        assert loaded == 2
        assert any("cublas" in p for p in loaded_paths)
        assert any("cudnn" in p for p in loaded_paths)

    def test_default_system_uses_platform_system(self, monkeypatch):
        """When system arg is None, fall back to platform.system()."""
        import transcriber

        monkeypatch.setattr(transcriber.platform, "system", lambda: "Linux")
        loaded = _preload_cuda_libs(lib_dirs=[], cdll=lambda _p: None)
        assert loaded == 0

        monkeypatch.setattr(transcriber.platform, "system", lambda: "Darwin")
        # If platform is Darwin, it should not even call cdll.
        called = []
        loaded = _preload_cuda_libs(
            lib_dirs=[Path("/nonexistent")],
            cdll=lambda p: called.append(p),
        )
        assert loaded == 0
        assert called == []


# ---------- _discover_nvidia_lib_dirs ----------


class TestDiscoverNvidiaLibDirs:
    def test_returns_empty_when_packages_missing(self, monkeypatch):
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("nvidia."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "nvidia.cublas", raising=False)
        monkeypatch.delitem(sys.modules, "nvidia.cudnn", raising=False)

        assert _discover_nvidia_lib_dirs() == []

    def test_finds_lib_subdir_under_pkg_path(self, monkeypatch, tmp_path):
        import sys
        import types

        cublas_root = tmp_path / "nvidia_cublas"
        (cublas_root / "lib").mkdir(parents=True)
        (cublas_root / "lib" / "libcublas.so.12").write_bytes(b"")

        fake_pkg = types.ModuleType("nvidia.cublas")
        fake_pkg.__path__ = [str(cublas_root)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nvidia.cublas", fake_pkg)

        # Make nvidia.cudnn fail to import so only cublas is found.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nvidia.cudnn":
                raise ImportError("simulated")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "nvidia.cudnn", raising=False)

        result = _discover_nvidia_lib_dirs()
        assert result == [cublas_root / "lib"]

    def test_skips_pkg_when_lib_subdir_missing(self, monkeypatch, tmp_path):
        import builtins
        import sys
        import types

        empty_root = tmp_path / "nvidia_cublas_empty"
        empty_root.mkdir()  # no lib/ subdir

        fake_pkg = types.ModuleType("nvidia.cublas")
        fake_pkg.__path__ = [str(empty_root)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nvidia.cublas", fake_pkg)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nvidia.cudnn":
                raise ImportError("simulated")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "nvidia.cudnn", raising=False)

        assert _discover_nvidia_lib_dirs() == []


# ---------- _default_cdll_loader smoke ----------


class TestDefaultCdllLoader:
    def test_returns_ctypes_cdll_object(self, monkeypatch):
        """The default cdll loader should call ctypes.CDLL with RTLD_GLOBAL."""
        import ctypes

        from transcriber import _default_cdll_loader

        captured = {}

        class FakeCDLL:
            def __init__(self, path, mode):
                captured["path"] = path
                captured["mode"] = mode

        monkeypatch.setattr(ctypes, "CDLL", FakeCDLL)
        instance = _default_cdll_loader("/fake/libcublas.so.12")
        assert isinstance(instance, FakeCDLL)
        assert captured["path"] == "/fake/libcublas.so.12"
        assert captured["mode"] == ctypes.RTLD_GLOBAL
