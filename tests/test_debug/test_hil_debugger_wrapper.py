# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HIL debugger wrapper tests

"""Tests for HILDebugger delegation to the telemetry server daemon."""

from __future__ import annotations

import pytest

from sc_neurocore.debug import hil_debugger


def test_hil_debugger_delegates_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[object] = []

    class Daemon:
        def __init__(self, port: int) -> None:
            self.port = port
            self.is_running = False
            self.started = 0
            self.stopped = 0
            instances.append(self)

        def start(self) -> bool:
            self.started += 1
            self.is_running = True
            return True

        def stop(self) -> None:
            self.stopped += 1
            self.is_running = False

    monkeypatch.setattr(hil_debugger, "HILServerDaemon", Daemon)

    debugger = hil_debugger.HILDebugger(port=8124)

    assert instances == [debugger.daemon]
    assert debugger.url == "http://localhost:8124"
    assert debugger.is_running is False
    assert debugger.start() is True
    assert debugger.daemon.started == 1
    assert debugger.is_running is True
    debugger.stop()
    assert debugger.daemon.stopped == 1
    assert debugger.is_running is False
