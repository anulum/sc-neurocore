# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ButeraRespiratoryNeuron

"""Full pipeline test for ButeraRespiratoryNeuron (Butera, Rinzel & Smith 1999).

Pre-Bötzinger respiratory neuron with persistent Na⁺ current and
slow h_nap inactivation. Bursting at high current."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestButeraIsolation:
    def test_construction(self):
        n = ButeraRespiratoryNeuron()
        assert n.v == -50.0
        assert n.h_nap == 0.5

    def test_step_returns_binary(self):
        n = ButeraRespiratoryNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(10.0) for _ in range(10_000))
        assert spikes == 0

    def test_spikes_at_high_current(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(100.0) for _ in range(100_000))
        assert spikes > 100, f"too few spikes at I=100: {spikes}"

    def test_persistent_na_inactivation(self):
        """h_nap should change from initial value under sustained drive."""
        n = ButeraRespiratoryNeuron()
        h_init = n.h_nap
        for _ in range(100_000):
            n.step(100.0)
        assert n.h_nap != h_init

    def test_numerical_stability(self):
        for I in [0, 10, 50, 100]:
            n = ButeraRespiratoryNeuron()
            for _ in range(50_000):
                n.step(float(I))
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.n), f"n NaN at I={I}"
            assert np.isfinite(n.h_nap), f"h_nap NaN at I={I}"

    def test_gating_bounded(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(50_000):
            n.step(100.0)
        assert 0 <= n.n <= 1
        assert 0 <= n.h_nap <= 1

    def test_reset(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(1000):
            n.step(100.0)
        n.reset()
        assert n.v == -50.0
        assert n.n == 0.01
        assert n.h_nap == 0.5


class TestButeraNetwork:
    def test_population(self):
        pop = Population(ButeraRespiratoryNeuron, n=5, label="butera")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestButeraAnalysis:
    def _get_train(self):
        n = ButeraRespiratoryNeuron()
        train = np.zeros(100_000, dtype=np.int8)
        for t in range(100_000):
            train[t] = n.step(100.0)
        return train

    def test_firing_rate(self):
        train = self._get_train()
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_train()
        assert spike_count(train) > 100

    def test_isi(self):
        train = self._get_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
