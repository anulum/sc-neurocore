# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMapper from former test_hardware.py

"""Focused suite: TestMapper from former test_hardware.py."""

from __future__ import annotations

from tests.hardware_support import *  # noqa: F403

class TestMapper:
    def _make_adj(self, n=20, density=0.1, seed=42):
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) < density).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def test_greedy_no_collisions(self):
        adj = self._make_adj()
        mapper = Mapper()
        placements = mapper.map_greedy(adj, get_device(DeviceFamily.LOIHI))
        neuron_ids = [p.neuron_id for p in placements]
        assert len(set(neuron_ids)) == len(neuron_ids)

    def test_balanced_all_placed(self):
        adj = self._make_adj()
        mapper = Mapper()
        placements = mapper.map_balanced(adj, get_device(DeviceFamily.LOIHI))
        assert len(placements) == 20

    def test_locality_clusters_neighbors(self):
        n = 20
        adj = np.zeros((n, n))
        # Create two clusters: 0-9 and 10-19
        for i in range(9):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        for i in range(10, 19):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        mapper = Mapper()
        # Use FPGA with small cores to force splitting
        device = DeviceSpec(
            family=DeviceFamily.FPGA_GENERIC,
            cores=10,
            neurons_per_core=10,
            synapses_per_core=1000,
            axons_per_core=100,
            tick_ns=100,
            precision_bits=16,
            supports_learning=True,
            power_per_core_mw=1.0,
        )
        placements = mapper.map_locality(adj, device)
        assert len(placements) == n
        # Check cluster 0 neurons are mostly on same core
        cluster0_cores = {placements[i].core_id for i in range(10)}
        assert len(cluster0_cores) <= 2  # should be 1 or 2 cores

    def test_greedy_core_ids_valid(self):
        adj = self._make_adj(100)
        device = get_device(DeviceFamily.LOIHI)
        mapper = Mapper()
        placements = mapper.map_greedy(adj, device)
        for p in placements:
            assert p.core_id >= 0
            assert p.local_id >= 0

    def test_balanced_mapping_caps_core_count_to_device_limit(self):
        adj = self._make_adj(n=10, density=0.2)
        mapper = Mapper()
        device = DeviceSpec(
            family=DeviceFamily.FPGA_GENERIC,
            cores=2,
            neurons_per_core=3,
            synapses_per_core=10_000,
            axons_per_core=10_000,
            tick_ns=100.0,
            precision_bits=16,
            supports_learning=True,
            power_per_core_mw=1.0,
        )
        placements = mapper.map_balanced(adj, device)
        assert len(placements) == 10
        assert {p.core_id for p in placements} <= {0, 1}

    def test_locality_mapping_fallback_places_all_remaining_neurons(self):
        n = 9
        adj = np.zeros((n, n), dtype=float)
        mapper = Mapper()
        device = DeviceSpec(
            family=DeviceFamily.FPGA_GENERIC,
            cores=1,
            neurons_per_core=2,
            synapses_per_core=10_000,
            axons_per_core=10_000,
            tick_ns=100.0,
            precision_bits=16,
            supports_learning=True,
            power_per_core_mw=1.0,
        )
        placements = mapper.map_locality(adj, device)
        assert len(placements) == n
        assert sorted(p.neuron_id for p in placements) == list(range(n))
