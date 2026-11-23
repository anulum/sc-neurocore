# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTiming from former test_chiplet_routing.py

"""Focused suite: TestTiming from former test_chiplet_routing.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    simulate_timing,
)


class TestTiming:
    """Lowest-latency path aggregation."""

    def test_same_die(self) -> None:
        """Timing a path to the same die gives zero latency."""
        result = simulate_timing(ChipletTopology.ring(3), 0, 0)
        assert result is not None and result.total_latency_ns == 0.0

    def test_adjacent_and_multihop_paths(self) -> None:
        """Timing follows the direct path to a neighbour and a multi-hop path across a mesh."""
        adjacent = simulate_timing(ChipletTopology.ring(4), 0, 1)
        multihop = simulate_timing(ChipletTopology.mesh_2d(2, 3), 0, 5)
        assert adjacent is not None and adjacent.path == [0, 1]
        assert multihop is not None and len(multihop.path) > 2

    def test_unreachable_returns_none(self) -> None:
        """Timing an unreachable die returns nothing."""
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        assert simulate_timing(topology, 0, 1) is None

    def test_path_accumulates_bandwidth_jitter_and_ber(self) -> None:
        """A multi-hop path reports its narrowest bandwidth, largest jitter and worst error rate."""
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1), ChipletDie(2)])
        topology.add_link(InterposerLink(0, 1, bandwidth_gbps=100.0, jitter_ns=0.2))
        topology.add_link(InterposerLink(1, 2, bandwidth_gbps=10.0, bit_error_rate=1e-9))
        result = simulate_timing(topology, 0, 2)
        assert result is not None
        assert result.min_bandwidth_gbps == 10.0
        assert result.max_jitter_ns == 0.2
        assert result.worst_ber == 1e-9
