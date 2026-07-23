# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUCIePartitioning from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestUCIePartitioning from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403

class TestUCIePartitioning:
    """Chiplet die-to-die neuron array partitioning."""

    def test_basic_partition(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(1000, 0.1, tile_count=4)
        assert p.tile_count == 4
        assert p.neurons_per_tile == 250
        assert p.die_to_die_bandwidth_gbps >= 0

    def test_partition_map_covers_all_neurons(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=4)
        all_neurons = []
        for ids in p.partition_map.values():
            all_neurons.extend(ids)
        assert len(set(all_neurons)) == 100

    def test_more_tiles_more_inter_traffic(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p2 = advise_ucie_partition(1000, 0.1, tile_count=2)
        p8 = advise_ucie_partition(1000, 0.1, tile_count=8)
        # More tiles → more inter-tile fraction → more bandwidth
        assert p8.die_to_die_bandwidth_gbps >= p2.die_to_die_bandwidth_gbps

    def test_latency_scales_with_tiles(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p2 = advise_ucie_partition(100, 0.1, tile_count=2)
        p8 = advise_ucie_partition(100, 0.1, tile_count=8)
        assert p8.latency_penalty_ns > p2.latency_penalty_ns

    def test_single_tile_no_overhead(self):
        from sc_neurocore.compiler.intelligence import advise_ucie_partition

        p = advise_ucie_partition(100, 0.1, tile_count=1)
        assert p.inter_tile_spikes == 0
        assert p.latency_penalty_ns == 0
