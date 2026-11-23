# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlanarTopologies from former test_chiplet_topology.py

"""Focused suite: TestPlanarTopologies from former test_chiplet_topology.py."""

from __future__ import annotations

import pytest
from collections.abc import (
    Callable,
)
from sc_neurocore.chiplet import (
    ChipletTopology,
    InterposerTech,
    make_torus,
    simulate_timing,
)


class TestPlanarTopologies:
    """Mesh, ring, star, and torus graph contracts."""

    def test_mesh_has_expected_dies_links_and_unique_seeds(self) -> None:
        """A 2 by 3 mesh has six dies, seven links and unique seeds."""
        topology = ChipletTopology.mesh_2d(2, 3)
        assert topology.num_dies == 6
        assert len(topology.links) == 7
        assert len({die.lfsr_seed for die in topology.dies}) == 6

    def test_ring_lookup_contracts(self) -> None:
        """A ring's link and die lookups follow its order."""
        topology = ChipletTopology.ring(4, InterposerTech.EMIB)
        assert len(topology.links) == 4
        assert topology.get_links_from(0)[0].dst_die == 1
        assert topology.get_links_to(0)[0].src_die == 3
        assert topology.get_die(3) is not None
        assert topology.get_die(99) is None

    def test_star_is_bidirectional_through_hub(self) -> None:
        """A star links every leaf to the hub in both directions and routes leaf to leaf through it."""
        topology = ChipletTopology.star(5)
        assert len(topology.links) == 8
        assert len(topology.get_links_from(0)) == 4
        assert len(topology.get_links_to(0)) == 4
        timing = simulate_timing(topology, 1, 2)
        assert timing is not None and timing.path == [1, 0, 2]

    def test_torus_wraps_and_remains_connected(self) -> None:
        """A torus adds wrap-around links and stays connected."""
        topology = make_torus(2, 3)
        assert topology.num_dies == 6
        assert len(topology.links) == 12
        assert 0 in {link.dst_die for link in topology.get_links_from(2)}
        assert simulate_timing(topology, 0, 5) is not None

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: ChipletTopology.mesh_2d(0, 1),
            lambda: ChipletTopology.ring(0),
            lambda: ChipletTopology.star(0),
            lambda: make_torus(1, 0),
        ],
    )
    def test_empty_topology_factories_fail(self, factory: Callable[[], object]) -> None:
        """Each topology factory refuses an empty size."""
        with pytest.raises(ValueError):
            factory()
