# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC scope session edge tests

"""Contracts for SC scope session start, capture, and status edges."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.debug.sc_scope import (
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
)
from tests.test_debug.sc_scope_edges_support import _session


def test_hardware_transport_connects_and_reads_none() -> None:
    """A non-simulated transport connects optimistically but yields no bitstream data."""
    backend = TransportBackend(TransportConfig(TransportType.JTAG))

    assert backend.connect() is True
    assert backend.is_connected is True
    assert backend.read_bitstream(8) is None


def test_disconnected_transport_reads_none() -> None:
    """A transport that has not connected returns no data."""
    backend = TransportBackend(TransportConfig(TransportType.SIMULATED))

    assert backend.read_bitstream(8) is None


def test_scope_session_start_fails_when_transport_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport that refuses to connect makes the session fail to start."""
    session = _session()
    monkeypatch.setattr(session.transport, "connect", lambda: False)

    assert session.start() is False
    assert session.is_running is False


def test_capture_one_returns_none_when_transport_yields_no_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A running session returns no sample when the transport produces no words."""
    session = _session()
    assert session.start() is True
    monkeypatch.setattr(session.transport, "read_bitstream", lambda *_a, **_k: None)

    assert session.capture_one(layer_id=0) is None


def test_session_status_reports_zero_elapsed_before_start() -> None:
    """Status before any start reports a zero elapsed time."""
    session: ScopeSession = _session()

    status: dict[str, Any] = session.status()

    assert status["running"] is False
    assert status["elapsed_s"] == 0
