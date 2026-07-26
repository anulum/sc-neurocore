# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical-twin TCP tests

"""Connection, empty reply, malformed JSON, and successful exchange contracts."""

import pytest

from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge
from tests.physical_twin_support import _FakeSocket


def test_tcp_connection_failure_marks_disconnected(monkeypatch):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    def refuse(address, timeout):  # type: ignore[no-untyped-def] # Preserved legacy nested-helper AST
        raise OSError("connection refused")

    monkeypatch.setattr("socket.create_connection", refuse)
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ConnectionError, match="hardware twin connection failed"):
        bridge.sync_step(0.5, 1)
    assert bridge.connected is False


def test_tcp_empty_reply_marks_disconnected(monkeypatch):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    # makefile yields no line -> next() raises StopIteration inside sync.
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *a, **k: _FakeSocket(reply_lines=[]),  # type: ignore[no-untyped-call] # Legacy support
    )
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ConnectionError, match="closed connection without a reply"):
        bridge.sync_step(0.5, 1)
    assert bridge.connected is False


def test_tcp_non_json_reply_raises_value_error(monkeypatch):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *a, **k: _FakeSocket(reply_lines=["definitely not json\n"]),  # type: ignore[no-untyped-call] # Legacy support
    )
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ValueError, match="not valid JSON"):
        bridge.sync_step(0.5, 1)


def test_tcp_successful_exchange_marks_connected(monkeypatch):  # type: ignore[no-untyped-def] # Preserved legacy test AST
    sock = _FakeSocket(reply_lines=['{"v_mem": 0.5}\n'])  # type: ignore[no-untyped-call] # Legacy support
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: sock)
    bridge = PhysicalTwinBridge(mode="TCP")
    result = bridge.sync_step(0.5, 1)
    assert result == pytest.approx(0.5)
    assert bridge.connected is True
    # Software and hardware agree, so no divergence path is exercised here.
