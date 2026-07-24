# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNullMessageOptimizer from former test_twinsync.py

"""Focused suite: TestNullMessageOptimizer from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestNullMessageOptimizer:
    def test_safe_advance(self):
        nmo = NullMessageOptimizer(3)
        nmo.broadcast_null(0, 500)
        nmo.broadcast_null(1, 300)
        nmo.broadcast_null(2, 400)
        safe = nmo.safe_advance_time(0)
        assert safe == 1300  # min peer: node1 at 300+1000

    def test_lookahead_can_advance(self):
        lc = LookaheadConfig(0, lookahead_ns=500)
        lc.send_null_message(1000)
        assert lc.can_advance_to(1400) is True
        assert lc.can_advance_to(1600) is False

    def test_safe_advance_single_node_uses_own_horizon(self):
        # With no peers to constrain it, a node may advance to its own last
        # null-message time plus its lookahead horizon.
        nmo = NullMessageOptimizer(1, default_lookahead_ns=1000)
        nmo.broadcast_null(0, 500)
        assert nmo.safe_advance_time(0) == 1500
