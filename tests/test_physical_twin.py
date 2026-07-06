# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PhysicalTwinBridge (mock-based, no real FPGA)

"""Tests for PhysicalTwinBridge (mock-based, no real FPGA)."""

import numpy as np
import pytest

from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge


class _FakeSocket:
    """Minimal ``socket.create_connection`` context-manager stand-in."""

    def __init__(self, reply_lines):
        self._reply_lines = reply_lines
        self.sent: list[bytes] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def sendall(self, data):
        self.sent.append(data)

    def makefile(self, mode, encoding):
        assert mode == "r"
        assert encoding == "utf-8"
        return iter(self._reply_lines)


def test_import():
    assert PhysicalTwinBridge is not None


def test_instantiation_default():
    bridge = PhysicalTwinBridge()
    assert bridge.connected is True
    assert bridge.ip == "192.168.2.99"
    assert bridge.port == 5000


def test_instantiation_custom():
    bridge = PhysicalTwinBridge(ip="10.0.0.1", port=9999)
    assert bridge.ip == "10.0.0.1"
    assert bridge.port == 9999


def test_sync_step_returns_float():
    bridge = PhysicalTwinBridge()
    hw_v = bridge.sync_step(sw_v_mem=-65.0, sw_spike=0)
    assert isinstance(hw_v, (float, np.floating))


def test_sync_step_close_to_input():
    bridge = PhysicalTwinBridge()
    sw_v = -65.0
    results = [bridge.sync_step(sw_v, 0) for _ in range(100)]
    mean_diff = np.mean(np.abs(np.array(results) - sw_v))
    # Mock adds N(0, 0.01) noise, so mean abs diff << 0.1
    assert mean_diff < 0.05


def test_sync_step_disconnected_returns_input():
    bridge = PhysicalTwinBridge()
    bridge.connected = False
    sw_v = -70.0
    assert bridge.sync_step(sw_v, 1) == sw_v


def test_has_expected_attributes():
    bridge = PhysicalTwinBridge()
    assert hasattr(bridge, "sync_step")
    assert hasattr(bridge, "connected")
    assert hasattr(bridge, "ip")
    assert hasattr(bridge, "port")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="'EMULATION' or 'TCP'"):
        PhysicalTwinBridge(mode="BOGUS")


def test_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        PhysicalTwinBridge(timeout_s=0.0)


def test_rejects_negative_noise_sigma():
    with pytest.raises(ValueError, match="noise_sigma must be non-negative"):
        PhysicalTwinBridge(noise_sigma=-0.1)


def test_rejects_non_positive_divergence_threshold():
    with pytest.raises(ValueError, match="divergence_threshold must be positive"):
        PhysicalTwinBridge(divergence_threshold=0.0)


def test_tcp_connection_failure_marks_disconnected(monkeypatch):
    def refuse(address, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("socket.create_connection", refuse)
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ConnectionError, match="hardware twin connection failed"):
        bridge.sync_step(0.5, 1)
    assert bridge.connected is False


def test_tcp_empty_reply_marks_disconnected(monkeypatch):
    # makefile yields no line -> next() raises StopIteration inside sync.
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: _FakeSocket(reply_lines=[]))
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ConnectionError, match="closed connection without a reply"):
        bridge.sync_step(0.5, 1)
    assert bridge.connected is False


def test_tcp_non_json_reply_raises_value_error(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *a, **k: _FakeSocket(reply_lines=["definitely not json\n"]),
    )
    bridge = PhysicalTwinBridge(mode="TCP")
    with pytest.raises(ValueError, match="not valid JSON"):
        bridge.sync_step(0.5, 1)


def test_tcp_successful_exchange_marks_connected(monkeypatch):
    sock = _FakeSocket(reply_lines=['{"v_mem": 0.5}\n'])
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: sock)
    bridge = PhysicalTwinBridge(mode="TCP")
    result = bridge.sync_step(0.5, 1)
    assert result == pytest.approx(0.5)
    assert bridge.connected is True
    # Software and hardware agree, so no divergence path is exercised here.
