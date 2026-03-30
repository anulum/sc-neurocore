# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GutkinErmentroutNeuron

"""Full pipeline test for GutkinErmentroutNeuron (Gutkin & Ermentrout 1998).

Minimal 2D conductance model: persistent Na + K. Type-I excitability."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.gutkin_ermentrout import GutkinErmentroutNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGEIsolation:
    def test_construction(self):
        n = GutkinErmentroutNeuron()
        assert n.v == -65.0
        assert n.g_na == 20.0

    def test_step_returns_binary(self):
        assert GutkinErmentroutNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = GutkinErmentroutNeuron()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = GutkinErmentroutNeuron()
        assert sum(n.step(100.0) for _ in range(5000)) > 30

    def test_rate_increases_with_input(self):
        n_low = GutkinErmentroutNeuron()
        n_high = GutkinErmentroutNeuron()
        s_low = sum(n_low.step(50.0) for _ in range(5000))
        s_high = sum(n_high.step(150.0) for _ in range(5000))
        assert s_high > s_low

    def test_n_gating(self):
        """K gate n should change under drive."""
        n = GutkinErmentroutNeuron()
        n_init = n.n
        for _ in range(2000):
            n.step(100.0)
        assert n.n != n_init

    def test_persistent_na(self):
        """m_inf is instantaneous — no gating variable stored."""
        n = GutkinErmentroutNeuron()
        assert not hasattr(n, "m")

    def test_numerical_stability(self):
        for I in [0.0, 50.0, 100.0, 200.0]:
            n = GutkinErmentroutNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.n), f"n NaN at I={I}"

    def test_reset(self):
        n = GutkinErmentroutNeuron()
        for _ in range(2000):
            n.step(100.0)
        n.reset()
        assert n.v == -65.0
        assert n.n == 0.1

    def test_deterministic(self):
        n1 = GutkinErmentroutNeuron()
        n2 = GutkinErmentroutNeuron()
        for _ in range(500):
            assert n1.step(100.0) == n2.step(100.0)


class TestGENetwork:
    def test_population(self):
        assert Population(GutkinErmentroutNeuron, n=10, label="ge").n == 10

    def test_network_spikes(self):
        pop = Population(GutkinErmentroutNeuron, n=5, label="ge")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestGEAnalysis:
    def test_spike_count(self):
        n = GutkinErmentroutNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(100.0)
        assert spike_count(train) > 30
