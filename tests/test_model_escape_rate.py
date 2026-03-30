# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: EscapeRateNeuron

"""Full pipeline test for EscapeRateNeuron (Gerstner 2000).

Stochastic threshold — spike probability = ρ₀ exp((V-θ)/Δu)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestEscapeRateIsolation:
    def test_construction(self):
        n = EscapeRateNeuron()
        assert n.v == -70.0

    def test_step_returns_binary(self):
        assert EscapeRateNeuron().step(0.0) in (0, 1)

    def test_stochastic_spiking(self):
        """Should spike stochastically under sufficient drive."""
        n = EscapeRateNeuron()
        spikes = sum(n.step(50.0) for _ in range(10000))
        assert spikes > 10

    def test_rate_increases_with_input(self):
        n_low = EscapeRateNeuron()
        n_high = EscapeRateNeuron()
        s_low = sum(n_low.step(10.0) for _ in range(10000))
        s_high = sum(n_high.step(50.0) for _ in range(10000))
        assert s_high > s_low

    def test_safe_exp(self):
        """Extreme voltage should not overflow (safe_exp)."""
        n = EscapeRateNeuron()
        n.v = 1000.0
        result = n.step(0.0)
        assert result in (0, 1)
        assert np.isfinite(n.v)

    def test_state_finite(self):
        n = EscapeRateNeuron()
        for _ in range(10000):
            n.step(40.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = EscapeRateNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest


class TestEscapeRateNetwork:
    def test_population(self):
        assert Population(EscapeRateNeuron, n=10, label="esc").n == 10

    def test_network_spikes(self):
        pop = Population(EscapeRateNeuron, n=20, label="esc")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestEscapeRateAnalysis:
    def test_spike_count(self):
        n = EscapeRateNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(50.0)
        assert spike_count(train) > 10
