# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LapicqueNeuron

"""Full pipeline test for LapicqueNeuron (Lapicque 1907).

The original integrate-and-fire: tau dv/dt = -(v-v_rest) + R·I."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestLapicqueIsolation:
    def test_construction(self):
        n = LapicqueNeuron()
        assert n.v == 0.0
        assert n.tau == 20.0

    def test_step_returns_binary(self):
        assert LapicqueNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        """I < v_threshold/R = 1.0 → no spikes (steady state v < threshold)."""
        n = LapicqueNeuron()
        assert sum(n.step(0.5) for _ in range(1000)) == 0

    def test_spikes_above_rheobase(self):
        """I > v_threshold/R should produce spikes."""
        n = LapicqueNeuron()
        assert sum(n.step(2.0) for _ in range(1000)) > 10

    def test_rheobase(self):
        """Analytical rheobase: I_rh = v_threshold / R = 1.0."""
        n = LapicqueNeuron()
        s_below = sum(n.step(0.99) for _ in range(1000))
        n.reset()
        s_above = sum(n.step(1.5) for _ in range(1000))
        assert s_below == 0
        assert s_above > 0

    def test_rate_increases_with_input(self):
        n_low = LapicqueNeuron()
        n_high = LapicqueNeuron()
        s_low = sum(n_low.step(1.5) for _ in range(1000))
        s_high = sum(n_high.step(5.0) for _ in range(1000))
        assert s_high > s_low

    def test_voltage_clamp(self):
        """At steady state, v → R·I = I (since R=1). Below threshold."""
        n = LapicqueNeuron()
        for _ in range(500):
            n.step(0.5)
        assert abs(n.v - 0.5) < 0.1

    def test_hard_reset(self):
        n = LapicqueNeuron()
        for _ in range(100):
            if n.step(2.0):
                assert n.v == n.v_reset
                break

    def test_numerical_stability(self):
        for I in [0.0, 1.0, 5.0, 100.0]:
            n = LapicqueNeuron()
            for _ in range(1000):
                n.step(I)
            assert np.isfinite(n.v)

    def test_reset(self):
        n = LapicqueNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_rest

    def test_deterministic(self):
        n1 = LapicqueNeuron()
        n2 = LapicqueNeuron()
        for _ in range(200):
            assert n1.step(2.0) == n2.step(2.0)

    def test_custom_tau(self):
        """Larger tau → slower charging → fewer spikes."""
        n_fast = LapicqueNeuron(tau=5.0)
        n_slow = LapicqueNeuron(tau=50.0)
        s_fast = sum(n_fast.step(2.0) for _ in range(500))
        s_slow = sum(n_slow.step(2.0) for _ in range(500))
        assert s_fast > s_slow


class TestLapicqueNetwork:
    def test_population(self):
        assert Population(LapicqueNeuron, n=10, label="lap").n == 10

    def test_network_spikes(self):
        pop = Population(LapicqueNeuron, n=10, label="lap")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestLapicqueAnalysis:
    def test_spike_count(self):
        n = LapicqueNeuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(2.0)
        assert spike_count(train) > 10
