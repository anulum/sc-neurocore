# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMPITopology from former test_twinsync.py

"""Focused suite: TestMPITopology from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403

class TestMPITopology:
    def test_add_and_lookup(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "node0", neuron_range=(0, 100_000_000)))
        topo.add_rank(MPIRankMapping(1, "node0", neuron_range=(100_000_000, 200_000_000)))
        assert topo.num_ranks == 2
        assert topo.total_neurons == 200_000_000

    def test_rank_for_neuron(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "n0", neuron_range=(0, 1000)))
        topo.add_rank(MPIRankMapping(1, "n1", neuron_range=(1000, 2000)))
        assert topo.rank_for_neuron(500) == 0
        assert topo.rank_for_neuron(1500) == 1
        assert topo.rank_for_neuron(9999) is None

    def test_co_located(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "host_a", neuron_range=(0, 100)))
        topo.add_rank(MPIRankMapping(1, "host_a", neuron_range=(100, 200)))
        topo.add_rank(MPIRankMapping(2, "host_b", neuron_range=(200, 300)))
        assert topo.co_located_ranks(0) == [1]
        assert topo.co_located_ranks(2) == []
