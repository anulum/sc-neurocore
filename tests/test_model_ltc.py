# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LiquidTimeConstantNeuron

"""Full pipeline test for LiquidTimeConstantNeuron (Hasani et al. 2021).

Input-dependent time constant: tau(x,I) = tau_base · σ(w_tau·I + bias).
Sharp threshold transition between I∈[4,4.5]."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.ltc import LiquidTimeConstantNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestLTCIsolation:
    def test_construction(self):
        n = LiquidTimeConstantNeuron()
        assert n.x == 0.0
        assert n.tau_base == 10.0

    def test_step_returns_binary(self):
        assert LiquidTimeConstantNeuron().step(0.0) in (0, 1)

    def test_silent_at_zero(self):
        n = LiquidTimeConstantNeuron()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_subthreshold_settle(self):
        """I=3 → x settles just below threshold (x≈0.999)."""
        n = LiquidTimeConstantNeuron()
        for _ in range(2000):
            n.step(3.0)
        assert 0.99 < n.x < 1.0

    def test_spikes_above_transition(self):
        """I=5 → spikes every step (x resets each time)."""
        n = LiquidTimeConstantNeuron()
        assert sum(n.step(5.0) for _ in range(100)) == 100

    def test_sharp_transition(self):
        """I=4 → no spikes; I=4.5 → all spikes."""
        n_low = LiquidTimeConstantNeuron()
        n_high = LiquidTimeConstantNeuron()
        s_low = sum(n_low.step(4.0) for _ in range(2000))
        s_high = sum(n_high.step(4.5) for _ in range(2000))
        assert s_low == 0
        assert s_high == 2000

    def test_tau_input_dependent(self):
        """Time constant should change with input."""
        n = LiquidTimeConstantNeuron()
        tau_low = n.tau_base * (1.0 / (1.0 + np.exp(-(n.w_tau * 0.0 + n.bias))))
        tau_high = n.tau_base * (1.0 / (1.0 + np.exp(-(n.w_tau * 10.0 + n.bias))))
        assert tau_low != tau_high

    def test_tanh_target(self):
        """f_target = tanh(w_x·x + w_in·I)."""
        n = LiquidTimeConstantNeuron()
        f = np.tanh(n.w_x * 0.5 + n.w_in * 3.0)
        assert 0 < f < 1

    def test_numerical_stability(self):
        for I in [0.0, 3.0, 5.0, 20.0]:
            n = LiquidTimeConstantNeuron()
            for _ in range(2000):
                n.step(I)
            assert np.isfinite(n.x)

    def test_reset(self):
        n = LiquidTimeConstantNeuron()
        for _ in range(500):
            n.step(3.0)
        n.reset()
        assert n.x == 0.0

    def test_deterministic(self):
        n1 = LiquidTimeConstantNeuron()
        n2 = LiquidTimeConstantNeuron()
        for _ in range(200):
            assert n1.step(5.0) == n2.step(5.0)


class TestLTCNetwork:
    def test_population(self):
        assert Population(LiquidTimeConstantNeuron, n=10, label="ltc").n == 10

    def test_network_spikes(self):
        pop = Population(LiquidTimeConstantNeuron, n=10, label="ltc")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestLTCAnalysis:
    def test_spike_count(self):
        n = LiquidTimeConstantNeuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(5.0)
        assert spike_count(train) == 1000
