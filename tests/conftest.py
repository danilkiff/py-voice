"""Shared pytest fixtures: fakes for the faster-whisper model and friends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pytest


@dataclass
class FakeSegment:
    text: str


@dataclass
class FakeInfo:
    language: str = "ru"
    duration: float = 1.5


@dataclass
class FakeModel:
    """Records calls and returns canned segments/info."""

    segments: list[FakeSegment] = field(
        default_factory=lambda: [FakeSegment(text=" привет ")]
    )
    info: FakeInfo = field(default_factory=FakeInfo)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def transcribe(
        self, audio: str, **kwargs: Any
    ) -> tuple[Iterable[FakeSegment], FakeInfo]:
        self.calls.append((audio, kwargs))
        return iter(self.segments), self.info


@dataclass
class RecordingLoader:
    """Loader spy that returns the supplied model and counts invocations."""

    model: FakeModel
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def __call__(self, model_size: str, device: str, compute_type: str) -> FakeModel:
        self.calls.append((model_size, device, compute_type))
        return self.model


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def loader(fake_model: FakeModel) -> RecordingLoader:
    return RecordingLoader(model=fake_model)


@pytest.fixture
def audio_file(tmp_path):
    """A real (empty) file on disk that passes validate_audio_path."""
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF")
    return path


@pytest.fixture(autouse=True)
def _reset_transcriber_cache():
    """Make sure the module-level cache never leaks across tests."""
    from transcriber import clear_transcriber_cache

    clear_transcriber_cache()
    yield
    clear_transcriber_cache()
