# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network-topology optimiser contracts

"""Contracts for compiler network-topology optimisation."""

from __future__ import annotations


class TestNetworkTopology:
    def test_basic_partition(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.num_chips == 2
        assert len(plan.chip_assignment) == 5

    def test_all_intra(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1], 1: [0]}
        plan = optimize_network_topology(adj, num_chips=1)
        assert plan.inter_chip_spikes == 0

    def test_bandwidth_reduction(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.bandwidth_reduction >= 0.0

    def test_hub_neighbours_overflow_to_second_chip(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        # A hub fans out to three leaves but each chip holds only two neurons,
        # so two of the hub's edges are forced across the chip boundary.
        adj = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.inter_chip_spikes > 0
        assert plan.inter_chip_spikes + plan.intra_chip_spikes == 6
