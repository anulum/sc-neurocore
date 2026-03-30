# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MATNeuron

"""Full pipeline test for MATNeuron (Kobayashi 2009).

Multi-timescale Adaptive Threshold: theta = theta_base + theta1 + theta2.
Two adaptation time-scales (fast tau_1=10ms, slow tau_2=200ms)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMATIsolation:
    def test_construction(self):
        n = MATNeuron()
        assert n.v == -70.0
        assert n.theta1 == 0.0
        assert n.theta2 == 0.0

    def test_step_returns_binary(self):
        assert MATNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = MATNeuron()
        assert sum(n.step(15.0) for _ in range(2000)) == 0

    def test_spikes_under_drive(self):
        n = MATNeuron()
        assert sum(n.step(30.0) for _ in range(5000)) > 30

    def test_threshold_adaptation(self):
        """After spiking, theta1 + theta2 should raise the effective threshold."""
        n = MATNeuron()
        for _ in range(5000):
            if n.step(30.0):
                assert n.theta1 > 0
                assert n.theta2 > 0
                break

    def test_two_timescales(self):
        """theta1 decays faster than theta2."""
        n = MATNeuron()
        n.theta1 = 10.0
        n.theta2 = 10.0
        for _ in range(50):
            n.step(0.0)
        assert n.theta1 < n.theta2

    def test_adaptation_reduces_rate(self):
        """First half should have more spikes than second half (adaptation)."""
        n = MATNeuron()
        s1 = sum(n.step(40.0) for _ in range(2500))
        s2 = sum(n.step(40.0) for _ in range(2500))
        assert s1 >= s2

    def test_rate_increases_with_input(self):
        n_low = MATNeuron()
        n_high = MATNeuron()
        s_low = sum(n_low.step(25.0) for _ in range(5000))
        s_high = sum(n_high.step(50.0) for _ in range(5000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 30.0, 50.0, 100.0]:
            n = MATNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v)
            assert np.isfinite(n.theta1)
            assert np.isfinite(n.theta2)

    def test_reset(self):
        n = MATNeuron()
        for _ in range(2000):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta1 == 0.0
        assert n.theta2 == 0.0

    def test_deterministic(self):
        n1 = MATNeuron()
        n2 = MATNeuron()
        for _ in range(500):
            assert n1.step(30.0) == n2.step(30.0)


class TestMATNetwork:
    def test_population(self):
        assert Population(MATNeuron, n=10, label="mat").n == 10

    def test_network_spikes(self):
        pop = Population(MATNeuron, n=10, label="mat")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestMATAnalysis:
    def test_spike_count(self):
        n = MATNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(30.0)
        assert spike_count(train) > 30
