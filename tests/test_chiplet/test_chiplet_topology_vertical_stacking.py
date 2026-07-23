# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVerticalStacking from former test_chiplet_topology.py

"""Focused suite: TestVerticalStacking from former test_chiplet_topology.py."""

from __future__ import annotations

from chiplet_topology_support import *  # noqa: F403

class TestVerticalStacking:
    """TSV metadata and reciprocal topology insertion."""

    def test_tsv_unit_conversions(self) -> None:
        link = TSVLink(src_die=0, dst_die=1, tsv_count=1024, latency_ps=50.0)
        assert link.latency_ns == 0.05
        assert link.bandwidth_gbps > 100

    @pytest.mark.parametrize(
        ("stacking", "minimum_bandwidth"),
        [(StackingType.TSV_3D, 256.0), (StackingType.HYBRID_BONDING, 512.0)],
    )
    def test_stack_adds_reciprocal_links(
        self, stacking: StackingType, minimum_bandwidth: float
    ) -> None:
        topology = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
        link = add_3d_stack(topology, 0, 1, stacking)
        assert len(topology.links) == 2
        assert link.bandwidth_gbps >= minimum_bandwidth
        timing = simulate_timing(topology, 0, 1)
        assert timing is not None and timing.total_latency_ns < 0.1
