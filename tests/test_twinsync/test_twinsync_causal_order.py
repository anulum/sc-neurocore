# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalOrder from former test_twinsync.py

"""Focused suite: TestCausalOrder from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    TimeWarpEngine,
    TwinEvent,
)


class TestCausalOrder:
    """The processed log of each node stays in virtual-time order, and the verifier finds where it does not."""

    def test_ordered_events_no_violations(self) -> None:
        """Events processed in time order leave no violation."""
        eng = TimeWarpEngine(1)
        for t in [100, 200, 300]:
            eng.inject_event(TwinEvent(t, target_node=0, lamport_ts=t))
        for _ in range(3):
            eng.process_next()
        assert eng.verify_causal_order() == []

    def test_empty_no_violations(self) -> None:
        """An empty log has no violation."""
        eng = TimeWarpEngine(1)
        assert eng.verify_causal_order() == []

    def test_straggler_rollback_keeps_the_log_in_causal_order(self) -> None:
        """A straggler rolls the node back; the later event leaves the log and runs again after it."""
        eng = TimeWarpEngine(1)
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert [e.virtual_time_ns for e in eng.processed] == [100]
        eng.process_next()
        assert [e.virtual_time_ns for e in eng.processed] == [100, 200]
        assert eng.verify_causal_order() == []

    def test_a_log_out_of_causal_order_is_flagged(self) -> None:
        """The verifier flags neighbouring events of one node that run backwards in time."""
        eng = TimeWarpEngine(1)
        eng.processed = [TwinEvent(200, target_node=0), TwinEvent(100, target_node=0)]
        assert eng.verify_causal_order() == [(0, 1)]
