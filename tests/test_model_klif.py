# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: KLIFNeuron

"""Full pipeline test for KLIFNeuron.

LIF with learnable scaling factor k: V = alpha*V + k*I."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.klif import KLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestKLIFIsolation:
    def test_construction(self):
        n = KLIFNeuron()
        assert n.v == 0.0
        assert n.k == 1.0

    def test_step_returns_binary(self):
        assert KLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = KLIFNeuron()
        assert sum(n.step(0.05) for _ in range(10)) == 0

    def test_spikes_under_drive(self):
        n = KLIFNeuron()
        assert sum(n.step(0.5) for _ in range(100)) > 10

    def test_k_effect(self):
        """Higher k → stronger input → more spikes."""
        n_low = KLIFNeuron(k=0.5)
        n_high = KLIFNeuron(k=2.0)
        s_low = sum(n_low.step(0.3) for _ in range(500))
        s_high = sum(n_high.step(0.3) for _ in range(500))
        assert s_high > s_low

    def test_alpha_precomputed(self):
        n = KLIFNeuron()
        expected = np.exp(-1.0 / 10.0)
        assert abs(n.alpha - expected) < 1e-10

    def test_hard_reset(self):
        n = KLIFNeuron()
        for _ in range(100):
            if n.step(0.5):
                assert n.v == n.v_reset
                break

    def test_numerical_stability(self):
        for I in [0.0, 0.5, 1.0, 5.0]:
            n = KLIFNeuron()
            for _ in range(1000):
                n.step(I)
            assert np.isfinite(n.v)

    def test_reset(self):
        n = KLIFNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        n1 = KLIFNeuron()
        n2 = KLIFNeuron()
        for _ in range(200):
            assert n1.step(0.5) == n2.step(0.5)


class TestKLIFNetwork:
    def test_population(self):
        assert Population(KLIFNeuron, n=10, label="klif").n == 10

    def test_network_spikes(self):
        pop = Population(KLIFNeuron, n=10, label="klif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestKLIFAnalysis:
    def test_spike_count(self):
        n = KLIFNeuron()
        train = np.zeros(500, dtype=np.int8)
        for t in range(500):
            train[t] = n.step(0.5)
        assert spike_count(train) > 50
