# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for structured logging configuration."""

from __future__ import annotations

import io
import json
import logging

from sc_neurocore.utils.logging import configure_logging, JSONFormatter


def test_json_formatter_output():
    record = logging.LogRecord(
        name="sc_neurocore.test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    fmt = JSONFormatter()
    line = fmt.format(record)
    parsed = json.loads(line)
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed


def test_configure_json_mode():
    buf = io.StringIO()
    configure_logging(level="DEBUG", json=True, stream=buf)

    logger = logging.getLogger("sc_neurocore.test_json")
    logger.debug("structured test")

    output = buf.getvalue()
    parsed = json.loads(output.strip())
    assert parsed["msg"] == "structured test"
    assert parsed["level"] == "DEBUG"


def test_configure_human_mode():
    buf = io.StringIO()
    configure_logging(level="INFO", json=False, stream=buf)

    logger = logging.getLogger("sc_neurocore.test_human")
    logger.info("human test")

    output = buf.getvalue()
    assert "human test" in output
    assert "INFO" in output


def test_configure_clears_previous_handlers():
    buf = io.StringIO()
    configure_logging(level="INFO", stream=buf)
    configure_logging(level="INFO", stream=buf)

    root = logging.getLogger("sc_neurocore")
    assert len(root.handlers) == 1


def test_json_formatter_with_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="sc_neurocore.exc",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    fmt = JSONFormatter()
    line = fmt.format(record)
    parsed = json.loads(line)
    assert "boom" in parsed["exc"]
