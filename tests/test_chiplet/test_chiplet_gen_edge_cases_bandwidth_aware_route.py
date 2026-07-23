# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBandwidthAwareRoute from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestBandwidthAwareRoute from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403

class TestBandwidthAwareRoute:
    def test_visited_skip_and_queue_extension(self) -> None:
        # 4-die mesh 0↔1↔2↔3 + 1↔3 short-cut, all links 50 Gbps.
        topo = ChipletTopology()
        for i in range(4):
            topo.add_die(ChipletDie(die_id=i))
        for s, d in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)]:
            link = InterposerLink.from_tech(s, d, InterposerTech.UCIE)
            link.bandwidth_gbps = 50.0
            topo.add_link(link)
        # required 30 Gbps ≤ every link → BFS extends queue past die 1
        # (queue.append, line 1253). Two paths reach die 3 → visited-skip
        # at line 1247.
        path = bandwidth_aware_route(
            topo,
            src_die=0,
            dst_die=3,
            required_gbps=30.0,
        )
        assert path is not None
        assert path[0] == 0 and path[-1] == 3

    def test_returns_none_when_bandwidth_insufficient(self) -> None:
        topo = ChipletTopology()
        for i in range(2):
            topo.add_die(ChipletDie(die_id=i))
        link = InterposerLink.from_tech(0, 1, InterposerTech.UCIE)
        link.bandwidth_gbps = 10.0
        topo.add_link(link)
        # required 100 Gbps > 10 Gbps available → no path.
        path = bandwidth_aware_route(
            topo,
            src_die=0,
            dst_die=1,
            required_gbps=100.0,
        )
        assert path is None
