# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MorrisLecarNeuron

"""Full pipeline test for MorrisLecarNeuron (Morris & Lecar 1981).

Ca-K oscillator: 2D, tanh activation, Type-II excitability."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMLIsolation:
    def test_construction(self):
        n = MorrisLecarNeuron()
        assert n.v == -60.0
        assert n.g_ca == 4.0

    def test_step_returns_binary(self):
        assert MorrisLecarNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = MorrisLecarNeuron()
        assert sum(n.step(10.0) for _ in range(5000)) == 0

    def test_spikes_under_drive(self):
        n = MorrisLecarNeuron()
        assert sum(n.step(100.0) for _ in range(10000)) > 10

    def test_type_ii_non_monotonic(self):
        """ML is Type-II: oscillation band, high I reduces rate."""
        n_mid = MorrisLecarNeuron()
        n_high = MorrisLecarNeuron()
        s_mid = sum(n_mid.step(100.0) for _ in range(10000))
        s_high = sum(n_high.step(200.0) for _ in range(10000))
        assert s_mid > s_high

    def test_tanh_activation(self):
        n = MorrisLecarNeuron()
        m = n._m_inf(-60.0)
        assert 0.0 < m < 1.0

    def test_w_recovery(self):
        n = MorrisLecarNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert n.w != 0.0

    def test_lambda_rate(self):
        """lambda(v) = phi * cosh(...) should be positive."""
        n = MorrisLecarNeuron()
        assert n._lam(-60.0) > 0.0
        assert n._lam(0.0) > 0.0

    def test_numerical_stability(self):
        for I in [0.0, 50.0, 100.0, 200.0]:
            n = MorrisLecarNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.w), f"w NaN at I={I}"

    def test_bounded_orbit(self):
        n = MorrisLecarNeuron()
        for _ in range(10000):
            n.step(100.0)
        assert -100.0 < n.v < 150.0

    def test_reset(self):
        n = MorrisLecarNeuron()
        for _ in range(3000):
            n.step(100.0)
        n.reset()
        assert n.v == -60.0
        assert n.w == 0.0

    def test_deterministic(self):
        n1 = MorrisLecarNeuron()
        n2 = MorrisLecarNeuron()
        for _ in range(500):
            assert n1.step(100.0) == n2.step(100.0)


class TestMLNetwork:
    def test_population(self):
        assert Population(MorrisLecarNeuron, n=5, label="ml").n == 5

    def test_network_spikes(self):
        pop = Population(MorrisLecarNeuron, n=5, label="ml")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestMLAnalysis:
    def test_spike_count(self):
        n = MorrisLecarNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(100.0)
        assert spike_count(train) > 10
