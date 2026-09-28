"""Tests for the entry point and logging configuration."""

from __future__ import annotations

import json
import logging

import pytest
import uvicorn

from grounded.__main__ import main
from grounded.logging import JsonFormatter, configure_logging


def test_configure_logging_installs_json_handler() -> None:
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.DEBUG


def test_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=logging.sys.exc_info(),  # type: ignore[attr-defined]
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_main_logs_startup_and_serves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("APP_ENV", "ci")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    from grounded.config import get_settings

    served: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(app=app, **kw))
    get_settings.cache_clear()
    try:
        main()
    finally:
        get_settings.cache_clear()
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["msg"] == "started in ci"
    assert served["host"] == "127.0.0.1"  # loopback unless told otherwise
    assert served["port"] == 8000
