"""Tests for main.py — entry point."""

from __future__ import annotations

import main


class TestMain:
    def test_main_invokes_build_app_and_launch_on_all_interfaces(self, monkeypatch):
        launched = {}

        class FakeApp:
            def launch(self, **kwargs):
                launched.update(kwargs)

        monkeypatch.setattr(main, "build_app", lambda: FakeApp())
        main.main()
        assert launched == {"server_name": "0.0.0.0"}
