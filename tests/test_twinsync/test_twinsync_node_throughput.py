# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNodeThroughput from former test_twinsync.py

"""Focused suite: TestNodeThroughput from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    TimeWarpEngine,
    TwinEvent,
)


class TestNodeThroughput:
    """Processed-event counts per node."""

    def test_throughput_initial(self) -> None:
        """Every node starts with zero processed events."""
        eng = TimeWarpEngine(3)
        tp = eng.node_throughput()
        assert all(v == 0 for v in tp.values())

    def test_throughput_after_events(self) -> None:
        """Each node counts only the events it processed."""
        eng = TimeWarpEngine(2)
        for t in range(5):
            eng.inject_event(TwinEvent(t * 100, target_node=0, lamport_ts=t))
        for _ in range(5):
            eng.process_next()
        tp = eng.node_throughput()
        assert tp[0] == 5
        assert tp[1] == 0
