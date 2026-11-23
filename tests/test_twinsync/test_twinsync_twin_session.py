# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTwinSession from former test_twinsync.py

"""Focused suite: TestTwinSession from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    TwinSession,
)


class TestTwinSession:
    """Running a digital-twin session."""

    def test_create(self) -> None:
        """A new session has its nodes and is not running."""
        ts = TwinSession(4)
        assert ts.num_nodes == 4
        assert ts.running is False

    def test_start_stop(self) -> None:
        """Start and stop switch the running flag."""
        ts = TwinSession(2)
        ts.start()
        assert ts.running is True
        ts.stop()
        assert ts.running is False

    def test_inject_physical(self) -> None:
        """An injected physical event is counted."""
        ts = TwinSession(2)
        ts.inject_physical_event(1000, neuron_id=42, target_node=0)
        assert ts.physical_events_in == 1

    def test_advance(self) -> None:
        """Advancing processes the two injected events."""
        ts = TwinSession(2)
        ts.inject_physical_event(100, neuron_id=0, target_node=0)
        ts.inject_physical_event(200, neuron_id=1, target_node=1)
        processed = ts.advance(5)
        assert processed == 2

    def test_divergence_update(self) -> None:
        """Different spike rates give a positive divergence."""
        ts = TwinSession(1)
        dm = ts.update_divergence(10.0, 8.0, 0.5)
        assert dm.spike_rate_divergence > 0

    def test_status(self) -> None:
        """A running session reports its engine and optimistic mode."""
        ts = TwinSession(2)
        ts.start()
        st = ts.status()
        assert st["running"] is True
        assert "engine" in st
        assert st["mode"] == "optimistic"
