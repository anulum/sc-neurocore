# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChialvoNetwork from former test_model_chialvo_map.py

"""Focused suite: TestChialvoNetwork from former test_model_chialvo_map.py."""

from __future__ import annotations

from tests.model_chialvo_map_support import *  # noqa: F403

class TestChialvoNetwork:
    def test_population(self):
        pop = Population(ChialvoMapNeuron, n=10, label="chialvo")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(ChialvoMapNeuron, n=20, label="chialvo")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.1, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(ChialvoMapNeuron, n=10, label="chialvo")
        proj = Projection(pop, pop, weight=0.01, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.05, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)
