# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GalvesLocherbachNeuron

"""Full pipeline test for GalvesLocherbachNeuron (Galves & Löcherbach 2013).

Stochastic point process: P(spike) = σ(steepness·(V-threshold_rate)).
No ODE — purely probabilistic spiking with leaky integration."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.galves_locherbach import GalvesLocherbachNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGLIsolation:
    def test_construction(self):
        n = GalvesLocherbachNeuron()
        assert n.v == 0.0
        assert n.decay == 0.95

    def test_step_returns_binary(self):
        assert GalvesLocherbachNeuron().step(0.0) in (0, 1)

    def test_stochastic_spiking(self):
        """Should spike stochastically under drive."""
        n = GalvesLocherbachNeuron()
        spikes = sum(n.step(1.0) for _ in range(10000))
        assert spikes > 100

    def test_rate_increases_with_input(self):
        n_low = GalvesLocherbachNeuron()
        n_high = GalvesLocherbachNeuron()
        s_low = sum(n_low.step(0.1) for _ in range(5000))
        s_high = sum(n_high.step(2.0) for _ in range(5000))
        assert s_high > s_low

    def test_sigmoid_probability(self):
        """Firing probability should be sigmoid of v."""
        n = GalvesLocherbachNeuron()
        n.v = 10.0
        assert n._firing_prob() > 0.99
        n.v = -10.0
        assert n._firing_prob() < 0.01

    def test_decay(self):
        """Voltage should decay toward v_rest without input."""
        n = GalvesLocherbachNeuron()
        n.v = 5.0
        np.random.seed(0)
        for _ in range(100):
            n.step(0.0)
        assert abs(n.v) < 5.0

    def test_reset_on_spike(self):
        """After spiking, v should reset to v_rest."""
        n = GalvesLocherbachNeuron()
        n.v = 100.0
        np.random.seed(42)
        result = n.step(0.0)
        if result == 1:
            assert n.v == n.v_rest

    def test_numerical_stability(self):
        for I in [0.0, 0.5, 1.0, 5.0]:
            n = GalvesLocherbachNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"

    def test_reset(self):
        n = GalvesLocherbachNeuron()
        for _ in range(200):
            n.step(1.0)
        n.reset()
        assert n.v == n.v_rest

    def test_custom_steepness(self):
        """Higher steepness = sharper sigmoid transition."""
        n = GalvesLocherbachNeuron(steepness=20.0)
        n.v = 1.0
        p = n._firing_prob()
        assert p > 0.99

    def test_low_drive_few_spikes(self):
        """Near-zero input should produce few spikes."""
        n = GalvesLocherbachNeuron()
        s = sum(n.step(0.01) for _ in range(5000))
        assert s < 3000


class TestGLNetwork:
    def test_population(self):
        assert Population(GalvesLocherbachNeuron, n=10, label="gl").n == 10

    def test_network_spikes(self):
        pop = Population(GalvesLocherbachNeuron, n=10, label="gl")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestGLAnalysis:
    def test_spike_count(self):
        n = GalvesLocherbachNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(1.0)
        assert spike_count(train) > 100
