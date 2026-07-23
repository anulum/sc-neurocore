# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimeWarpEngine from former test_twinsync.py

"""Focused suite: TestTimeWarpEngine from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403

class TestTimeWarpEngine:
    def test_create(self):
        eng = TimeWarpEngine(4)
        assert len(eng.nodes) == 4

    def test_inject_and_process(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, event_type=EventType.SPIKE, target_node=0))
        ev = eng.process_next()
        assert ev is not None
        assert ev.virtual_time_ns == 100

    def test_ordering(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(300, target_node=0))
        eng.inject_event(TwinEvent(100, target_node=0))
        eng.inject_event(TwinEvent(200, target_node=0))
        ev1 = eng.process_next()
        ev2 = eng.process_next()
        assert ev1.virtual_time_ns == 100
        assert ev2.virtual_time_ns == 200

    def test_rollback_on_straggler(self):
        eng = TimeWarpEngine(2, checkpoint_interval_ns=1)
        # Process forward
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        # Inject straggler
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert eng.total_rollbacks > 0

    def test_identity_preserved_across_rollback(self):
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        eng.nodes[0].identity_deep = 0.42
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert eng.nodes[0].identity_deep == 0.42

    def test_gvt(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0))
        eng.inject_event(TwinEvent(200, target_node=1))
        eng.process_next()
        eng.process_next()
        gvt = eng.compute_gvt()
        assert gvt >= 0

    def test_fossil_collect(self):
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        for t in range(10):
            eng.inject_event(TwinEvent(t * 100, target_node=0, lamport_ts=t))
            eng.process_next()
        removed = eng.fossil_collect()
        assert removed >= 0

    def test_status(self):
        eng = TimeWarpEngine(2)
        st = eng.status()
        assert "num_nodes" in st
        assert "gvt_ns" in st

    def test_process_cancelled_event_short_circuits(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0, cancelled=True))
        ev = eng.process_next()
        assert ev is not None
        assert ev.cancelled is True
        assert eng.nodes[0].processed_events == 0  # not applied to the node

    def test_process_event_for_unknown_target(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=99))
        ev = eng.process_next()
        assert ev is not None
        assert ev.target_node == 99

    def test_process_event_merges_vector_clock(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0, vector_ts=np.array([3, 0])))
        eng.process_next()
        assert eng.nodes[0].vector_clock.clock[0] >= 3

    def test_rollback_restores_earlier_checkpoint(self):
        # A straggler with a checkpoint at or before its time rolls the node
        # back to that checkpoint (restoring lamport/vector state) before
        # re-advancing, rather than falling back to the bare target time.
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        eng.inject_event(TwinEvent(50, target_node=0, lamport_ts=1, vector_ts=np.array([1])))
        eng.process_next()  # checkpoint at vt=50
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=2, vector_ts=np.array([2])))
        eng.process_next()  # checkpoint at vt=200
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=3))
        eng.process_next()  # straggler -> rollback to cp@50
        assert eng.total_rollbacks > 0
        assert eng.nodes[0].local_virtual_time_ns == 100
