# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERRouterStats from former test_bridges_aer_router.py

"""Focused suite: TestAERRouterStats from former test_bridges_aer_router.py."""

from __future__ import annotations

from tests.bridges_aer_router_support import *  # noqa: F403


class TestAERRouterStats:
    """RouteStats correctness."""

    def test_get_stats_unknown_neuron_returns_none(self):
        router = AERRouter()
        assert router.get_stats(neuron_id=999) is None

    def test_stats_are_copies(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        s1 = router.get_stats(neuron_id=1)
        s2 = router.get_stats(neuron_id=1)
        assert s1 is not s2

    def test_fresh_stats_are_zero(self):
        router = AERRouter()
        router.register_route(neuron_id=1, addr="h:5000")
        stats = router.get_stats(neuron_id=1)
        assert stats.dispatched == 0
        assert stats.acked == 0
        assert stats.dropped == 0

    def test_route_stats_dataclass(self):
        rs = RouteStats(dispatched=10, acked=8, dropped=2)
        assert rs.dispatched == 10
        assert rs.acked == 8
        assert rs.dropped == 2
