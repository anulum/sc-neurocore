# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AdaptiveThresholdIFNeuron

"""Full pipeline test for AdaptiveThresholdIFNeuron (Platkiewicz & Bhatt 2010).

Verifies: import → isolation → Population → Projection → Network →
SpikeMonitor → analysis toolkit → reset. No shortcuts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestAdaptiveThresholdIFIsolation:
    """Model works in isolation."""

    def test_construction(self):
        n = AdaptiveThresholdIFNeuron()
        assert n.v == -65.0
        assert n.theta == -50.0

    def test_step_returns_binary(self):
        n = AdaptiveThresholdIFNeuron()
        result = n.step(0.0)
        assert result in (0, 1)

    def test_spikes_under_drive(self):
        n = AdaptiveThresholdIFNeuron()
        spikes = sum(n.step(100.0) for _ in range(2000))
        assert spikes > 0, "no spikes at I=100"

    def test_threshold_adapts(self):
        n = AdaptiveThresholdIFNeuron()
        theta_init = n.theta
        for _ in range(2000):
            n.step(100.0)
        assert n.theta > theta_init, "threshold did not increase after spiking"

    def test_state_finite(self):
        n = AdaptiveThresholdIFNeuron()
        for _ in range(5000):
            n.step(200.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.theta)

    def test_reset(self):
        n = AdaptiveThresholdIFNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_rest


class TestAdaptiveThresholdIFNetwork:
    """Model works in the full SC-NeuroCore network pipeline."""

    def test_population_creation(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        assert pop.n == 10
        assert pop.model_name == "AdaptiveThresholdIFNeuron"

    def test_network_produces_spikes(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=20, label="atif")
        proj = Projection(pop, pop, weight=1.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert len(trains) > 0, "no spike trains recorded"


class TestAdaptiveThresholdIFAnalysis:
    """Analysis toolkit works on spikes from this model."""

    def _get_binary_train(self):
        n = AdaptiveThresholdIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(80.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)  # dt=0.1ms (model dt)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) > 0

    def test_isi(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(intervals > 0)
            assert np.all(np.isfinite(intervals))
