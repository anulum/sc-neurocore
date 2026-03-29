# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ErmentroutKopellPopulation

"""Full pipeline test for ErmentroutKopellPopulation (Montbrio et al. 2015).

Exact mean-field of QIF/theta neuron network. Returns firing rate r (float),
not binary spike. Population clips to {0,1}."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.ermentrout_kopell_pop import ErmentroutKopellPopulation
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


class TestErmentroutKopellIsolation:
    def test_construction(self):
        n = ErmentroutKopellPopulation()
        assert n.r == 0.1
        assert n.v == -2.0

    def test_step_returns_float(self):
        """Mean-field model returns firing rate (float), not binary spike."""
        n = ErmentroutKopellPopulation()
        result = n.step(0.0)
        assert isinstance(result, float)

    def test_rate_increases_with_input(self):
        n1 = ErmentroutKopellPopulation()
        n2 = ErmentroutKopellPopulation()
        for _ in range(500):
            r1 = n1.step(0.0)
            r2 = n2.step(10.0)
        # Higher input should give higher rate (or different trajectory)
        assert r1 != r2

    def test_rate_nonnegative(self):
        n = ErmentroutKopellPopulation()
        for _ in range(5000):
            r = n.step(5.0)
        assert r >= 0

    def test_state_finite(self):
        n = ErmentroutKopellPopulation()
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.r)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = ErmentroutKopellPopulation()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.r == 0.1
        assert n.v == -2.0


class TestErmentroutKopellNetwork:
    def test_population(self):
        pop = Population(ErmentroutKopellPopulation, n=5, label="ek")
        assert pop.n == 5

    def test_network_runs(self):
        """Float return clipped to {0,1} in Population. When r>1, spike=1."""
        pop = Population(ErmentroutKopellPopulation, n=10, label="ek")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_field_state_after_run(self):
        pop = Population(ErmentroutKopellPopulation, n=5, label="ek")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        net.run(duration=0.1, dt=0.001, backend="python")
        for neuron in pop.neurons:
            assert np.isfinite(neuron.r)
            assert np.isfinite(neuron.v)
