# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhysicalTwinBridge from former test_pynq_driver.py

"""Focused suite: TestPhysicalTwinBridge from former test_pynq_driver.py."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal
import numpy as np
import pytest
from sc_neurocore.drivers.physical_twin import (
    PhysicalTwinBridge,
)


class TestPhysicalTwinBridge:
    """PhysicalTwinBridge exposes honest emulation and explicit TCP modes."""

    def test_default_bridge_is_emulation_without_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The default bridge emulates and prints nothing."""
        bridge = PhysicalTwinBridge(seed=7)

        assert bridge.mode == "EMULATION"
        assert bridge.connected is True
        assert not hasattr(bridge, "_TODO" + "_HIL")
        assert capsys.readouterr().out == ""

    def test_emulation_mode_is_deterministic_and_uses_instance_rng(self) -> None:
        """Emulation is deterministic and draws from the instance generator."""
        a = PhysicalTwinBridge(seed=123)
        b = PhysicalTwinBridge(seed=123)

        seq_a = [a.sync_step(0.4, 1) for _ in range(8)]
        seq_b = [b.sync_step(0.4, 1) for _ in range(8)]

        assert seq_a == seq_b
        assert all(np.isfinite(value) for value in seq_a)

    def test_tcp_mode_uses_json_line_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TCP mode sends and reads one JSON line per step."""
        writes: list[bytes] = []

        class FakeSocket:
            def __enter__(self) -> FakeSocket:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
                return False

            def sendall(self, data: bytes) -> None:
                writes.append(data)

            def makefile(self, mode: str, encoding: str) -> Iterator[str]:
                assert mode == "r"
                assert encoding == "utf-8"
                return iter(['{"v_mem": 0.875}\n'])

        def fake_create_connection(address: tuple[str, int], timeout: float) -> FakeSocket:
            assert address == ("192.168.2.99", 5000)
            assert timeout == 0.25
            return FakeSocket()

        monkeypatch.setattr("socket.create_connection", fake_create_connection)
        bridge = PhysicalTwinBridge(mode="TCP", timeout_s=0.25)

        assert bridge.connected is False
        assert bridge.sync_step(0.5, 1) == pytest.approx(0.875)
        assert writes == [b'{"spike":1,"v_mem":0.5}\n']

    def test_tcp_mode_rejects_malformed_hardware_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hardware reply without v_mem is rejected."""

        class FakeSocket:
            def __enter__(self) -> FakeSocket:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
                return False

            def sendall(self, data: bytes) -> None:
                pass

            def makefile(self, mode: str, encoding: str) -> Iterator[str]:
                return iter(["{}\n"])

        monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: FakeSocket())
        bridge = PhysicalTwinBridge(mode="TCP")

        with pytest.raises(ValueError, match="missing numeric 'v_mem'"):
            bridge.sync_step(0.5, 0)
