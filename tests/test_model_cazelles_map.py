# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: CazellesMapNeuron

"""Full pipeline test for CazellesMapNeuron (Cazelles et al. 2001).

2D bursting map neuron. Logistic-like fast dynamics x, slow modulation y."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestCazellesIsolation:
    def test_construction(self):
        n = CazellesMapNeuron()
        assert n.x == 0.1
        assert n.y == 0.0

    def test_step_returns_binary(self):
        n = CazellesMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self):
        n = CazellesMapNeuron()
        spikes = sum(n.step(0.2) for _ in range(5000))
        assert spikes > 100

    def test_slow_variable_modulates(self):
        """y should change from initial under sustained drive."""
        n = CazellesMapNeuron()
        y_init = n.y
        for _ in range(1000):
            n.step(0.2)
        assert n.y != y_init

    def test_x_clipped(self):
        """x should stay in [-2, 2] (np.clip in step)."""
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(1.0)
        assert -2.0 <= n.x <= 2.0

    def test_state_finite(self):
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.5)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_reset(self):
        n = CazellesMapNeuron()
        for _ in range(100):
            n.step(0.2)
        n.reset()
        assert n.x == 0.1
        assert n.y == 0.0


class TestCazellesNetwork:
    def test_population(self):
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(CazellesMapNeuron, n=20, label="caz")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(CazellesMapNeuron, n=10, label="caz")
        proj = Projection(pop, pop, weight=0.05, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestCazellesAnalysis:
    def _get_train(self):
        n = CazellesMapNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(0.2)
        return train

    def test_firing_rate(self):
        train = self._get_train()
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_train()
        assert spike_count(train) > 100

    def test_isi(self):
        train = self._get_train()
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
