# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalOrder from former test_twinsync.py

"""Focused suite: TestCausalOrder from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestCausalOrder:
    def test_ordered_events_no_violations(self):
        eng = TimeWarpEngine(1)
        for t in [100, 200, 300]:
            eng.inject_event(TwinEvent(t, target_node=0, lamport_ts=t))
        for _ in range(3):
            eng.process_next()
        assert eng.verify_causal_order() == []

    def test_empty_no_violations(self):
        eng = TimeWarpEngine(1)
        assert eng.verify_causal_order() == []

    def test_straggler_processing_records_violation(self):
        # Processing a later event then an earlier one for the same node leaves
        # the processed log out of causal order, which the verifier flags.
        eng = TimeWarpEngine(1)
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert (0, 1) in eng.verify_causal_order()
