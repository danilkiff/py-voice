"""Pick the best device and compute_type for the host."""

from __future__ import annotations

from typing import Callable


def ctranslate2_cuda_available() -> bool:
    """Default CUDA probe — uses CTranslate2's own detector.

    CTranslate2 is already a transitive dependency of faster-whisper, so this
    avoids pulling in PyTorch just to check for a GPU. Returns False if the
    library is missing or if the runtime probe raises for any reason.
    """
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def pick_device_and_compute_type(
    cuda_probe: Callable[[], bool] = ctranslate2_cuda_available,
) -> tuple[str, str]:
    """Return (device, compute_type) suitable for the current host.

    The cuda_probe is injectable so tests can simulate both branches without
    requiring a real GPU.
    """
    if cuda_probe():
        return "cuda", "float16"
    return "cpu", "int8"
