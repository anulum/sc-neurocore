# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPathSelection from former test_chiplet_routing.py

"""Focused suite: TestPathSelection from former test_chiplet_routing.py."""

from __future__ import annotations

from chiplet_routing_support import *  # noqa: F403


class TestPathSelection:
    """Disjoint, congestion-aware, and bandwidth-aware route selection."""

    def test_disjoint_path_contracts(self) -> None:
        topology = make_torus(2, 2)
        assert find_disjoint_paths(topology, 0, 0) == [[0]]
        paths = find_disjoint_paths(topology, 0, 3, max_paths=2)
        assert paths and all(path[0] == 0 and path[-1] == 3 for path in paths)
        if len(paths) == 2:
            first = set(zip(paths[0], paths[0][1:]))
            second = set(zip(paths[1], paths[1][1:]))
            assert first.isdisjoint(second)

    def test_unreachable_and_zero_limit(self) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        assert find_disjoint_paths(topology, 0, 1) == []
        assert find_disjoint_paths(ChipletTopology.ring(2), 0, 1, max_paths=0) == []
        with pytest.raises(ValueError, match="max_paths"):
            find_disjoint_paths(ChipletTopology.ring(2), 0, 1, max_paths=-1)

    def test_adaptive_route_avoids_then_falls_back(self) -> None:
        topology = make_torus(2, 3)
        path = adaptive_route(
            topology,
            0,
            1,
            CongestionReport(utilisation={(0, 1): 0.95}),
            congestion_threshold=0.8,
        )
        assert path is not None and (0, 1) not in set(zip(path, path[1:]))
        fallback = adaptive_route(
            ChipletTopology.ring(2),
            0,
            1,
            CongestionReport(utilisation={(0, 1): 1.0}),
            congestion_threshold=0.0,
        )
        assert fallback == [0, 1]

    def test_bandwidth_route_success_failure_and_validation(self) -> None:
        topology = ChipletTopology.ring(3, InterposerTech.COWOS)
        assert bandwidth_aware_route(topology, 0, 0, 100.0) == [0]
        assert bandwidth_aware_route(topology, 0, 1, 50.0) == [0, 1]
        assert bandwidth_aware_route(topology, 0, 1, 500.0) is None
        with pytest.raises(ValueError, match="required_gbps"):
            bandwidth_aware_route(topology, 0, 1, -1.0)
        with pytest.raises(ValueError, match="congestion_threshold"):
            adaptive_route(topology, 0, 1, CongestionReport(), -1.0)
