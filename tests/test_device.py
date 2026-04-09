"""Tests for device.py."""

from __future__ import annotations

import sys
import types

from device import ctranslate2_cuda_available, pick_device_and_compute_type


class TestPickDeviceAndComputeType:
    def test_cuda_available_returns_cuda_float16(self):
        assert pick_device_and_compute_type(cuda_probe=lambda: True) == (
            "cuda",
            "float16",
        )

    def test_cuda_unavailable_returns_cpu_int8(self):
        assert pick_device_and_compute_type(cuda_probe=lambda: False) == (
            "cpu",
            "int8",
        )

    def test_default_probe_does_not_raise(self):
        # The real probe must always return a tuple, regardless of host.
        device, compute_type = pick_device_and_compute_type()
        assert device in {"cuda", "cpu"}
        assert compute_type in {"float16", "int8"}


class TestCtranslate2CudaAvailable:
    def test_returns_false_when_ctranslate2_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ctranslate2":
                raise ImportError("simulated: ctranslate2 not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "ctranslate2", raising=False)

        assert ctranslate2_cuda_available() is False

    def test_returns_true_when_ctranslate2_reports_cuda(self, monkeypatch):
        fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
        assert ctranslate2_cuda_available() is True

    def test_returns_false_when_ctranslate2_reports_no_cuda(self, monkeypatch):
        fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 0)
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
        assert ctranslate2_cuda_available() is False

    def test_returns_false_when_probe_raises(self, monkeypatch):
        def boom() -> int:
            raise RuntimeError("driver missing")

        fake_ct2 = types.SimpleNamespace(get_cuda_device_count=boom)
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
        assert ctranslate2_cuda_available() is False
