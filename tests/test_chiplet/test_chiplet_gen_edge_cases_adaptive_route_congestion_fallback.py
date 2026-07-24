# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveRouteCongestionFallback from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestAdaptiveRouteCongestionFallback from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403


class TestAdaptiveRouteCongestionFallback:
    def test_falls_back_when_all_routes_congested(self) -> None:
        # 3-die ring 0–1–2–0, every link saturated → primary BFS fails,
        # fallback (line 1155) ignores congestion and finds a path.
        topo = ChipletTopology()
        for i in range(3):
            topo.add_die(ChipletDie(die_id=i))
        for s, d in [(0, 1), (1, 2), (2, 0)]:
            topo.add_link(InterposerLink.from_tech(s, d, InterposerTech.UCIE))
        congestion = CongestionReport()
        for link in topo.links:
            congestion.utilisation[(link.src_die, link.dst_die)] = 1.0
        # Threshold 0.5 → primary excludes every link, fallback wins.
        path = adaptive_route(
            topo,
            src_die=0,
            dst_die=2,
            congestion=congestion,
            congestion_threshold=0.5,
        )
        assert path is not None
        assert path[0] == 0 and path[-1] == 2
