# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_chiplet_routing.py

"""Module-level tests from former test_chiplet_routing.py."""

from __future__ import annotations

from chiplet_routing_support import *  # noqa: F403

def test_routing_table_queries() -> None:
    table = RoutingTable(die_id=0)
    table.add_route(10, 1, 20)
    table.add_route(11, 2, 30)
    table.add_route(12, 1, 40)
    assert table.num_entries == 3
    assert len(table.routes_to_die(1)) == 2
    assert table.target_dies == [1, 2]
def test_decorrelation_seeds_are_unique_nonzero_and_bounded() -> None:
    seeds = compute_decorrelation_seeds(ChipletTopology.mesh_2d(4, 4))
    assert len(set(seeds.values())) == len(seeds)
    assert all(1 <= seed <= 65535 for seed in seeds.values())
