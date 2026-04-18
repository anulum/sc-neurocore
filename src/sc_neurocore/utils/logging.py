# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Structured logging configuration for SC-NeuroCore

"""Structured logging configuration for SC-NeuroCore.

Usage::

    from sc_neurocore.utils.logging import configure_logging

    # Human-readable (default)
    configure_logging(level="DEBUG")

    # JSON-structured for production log aggregators
    configure_logging(level="INFO", json=True)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import IO


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_HUMAN_FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"


def configure_logging(
    level: str | int = "WARNING",
    json: bool = False,  # noqa: A002 — shadows builtin intentionally for clean API
    stream: IO[str] | None = None,
) -> None:
    """Configure the ``sc_neurocore`` logger hierarchy.

    Parameters
    ----------
    level : str or int
        Log level name (``"DEBUG"``, ``"INFO"``, etc.) or numeric level.
    json : bool
        If *True*, use :class:`JSONFormatter` for machine-parseable output.
    stream
        Output stream. Defaults to ``sys.stderr``.
    """
    root = logging.getLogger("sc_neurocore")
    root.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    if json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FMT))

    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else getattr(logging, level.upper()))
