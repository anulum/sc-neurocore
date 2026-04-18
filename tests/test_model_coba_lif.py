# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: COBALIFNeuron

"""Full pipeline test for COBALIFNeuron (Destexhe et al. 2001).

Conductance-based LIF with excitatory/inhibitory synaptic conductances."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestCOBAIsolation:
    def test_construction(self):
        n = COBALIFNeuron()
        assert n.v == -65.0
        assert n.g_e == 0.0

    def test_step_returns_binary(self):
        n = COBALIFNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = COBALIFNeuron()
        spikes = sum(n.step(100.0) for _ in range(5000))
        assert spikes == 0

    def test_spikes_under_drive(self):
        n = COBALIFNeuron()
        spikes = sum(n.step(500.0) for _ in range(10000))
        assert spikes > 10

    def test_conductance_decay(self):
        """g_e should decay exponentially without input."""
        n = COBALIFNeuron()
        n.g_e = 10.0
        n.step(0.0)
        assert n.g_e < 10.0

    def test_excitatory_conductance_injection(self):
        """delta_ge should increase g_e."""
        n = COBALIFNeuron()
        n.step(0.0, delta_ge=5.0)
        assert n.g_e > 0

    def test_inhibitory_conductance_injection(self):
        """delta_gi should increase g_i."""
        n = COBALIFNeuron()
        n.step(0.0, delta_gi=3.0)
        assert n.g_i > 0

    def test_state_finite(self):
        n = COBALIFNeuron()
        for _ in range(10000):
            n.step(500.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.g_e)
        assert np.isfinite(n.g_i)

    def test_reset(self):
        n = COBALIFNeuron()
        n.step(500.0, delta_ge=5.0, delta_gi=3.0)
        n.reset()
        assert n.v == n.e_l
        assert n.g_e == 0.0
        assert n.g_i == 0.0


class TestCOBANetwork:
    def test_population(self):
        pop = Population(COBALIFNeuron, n=10, label="coba")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(COBALIFNeuron, n=20, label="coba")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(COBALIFNeuron, n=10, label="coba")
        proj = Projection(pop, pop, weight=50.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestCOBAAnalysis:
    def _get_train(self):
        n = COBALIFNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(600.0)
        return train

    def test_firing_rate(self):
        train = self._get_train()
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_train()
        assert spike_count(train) > 10

    def test_isi(self):
        train = self._get_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
