# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCazellesNetwork from former test_model_cazelles_map.py

"""Focused suite: TestCazellesNetwork from former test_model_cazelles_map.py."""

from __future__ import annotations

from tests.model_cazelles_map_support import *  # noqa: F403


class TestCazellesNetwork:
    def test_population(self):
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(CazellesMapNeuron, n=20, label="caz")
        mon = SpikeMonitor(pop)
        net = Network(pop, mon)
        net.run(duration=0.6, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        proj = Projection(pop, pop, weight=0.05, probability=0.3, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, mon)
        assert isinstance(mon.spike_trains, dict)
