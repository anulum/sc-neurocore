# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSyncBarrier from former test_twinsync.py

"""Focused suite: TestSyncBarrier from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    EventType,
    TimeWarpEngine,
)


class TestSyncBarrier:
    """Synchronisation barriers across all nodes."""

    def test_barrier_injects_to_all_nodes(self) -> None:
        """A barrier puts one event in the queue per node."""
        eng = TimeWarpEngine(4)
        eng.inject_sync_barrier(5000)
        assert len(eng.event_queue) == 4

    def test_barrier_processed(self) -> None:
        """Processed barrier events carry the barrier type."""
        eng = TimeWarpEngine(2)
        eng.inject_sync_barrier(1000)
        eng.process_next()
        eng.process_next()
        assert len(eng.processed) == 2
        for ev in eng.processed:
            assert ev.event_type == EventType.SYNC_BARRIER

    def test_barrier_advances_lvt(self) -> None:
        """A processed barrier moves each node to its time."""
        eng = TimeWarpEngine(2)
        eng.inject_sync_barrier(5000)
        eng.process_next()
        eng.process_next()
        for n in eng.nodes.values():
            assert n.local_virtual_time_ns == 5000
