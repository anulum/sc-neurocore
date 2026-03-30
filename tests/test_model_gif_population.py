# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GIFPopulationNeuron

"""Full pipeline test for GIFPopulationNeuron (Mensi et al. 2012).

Generalized IF with escape-rate stochastic threshold + spike-triggered
adaptation current eta."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGIFIsolation:
    def test_construction(self):
        n = GIFPopulationNeuron()
        assert n.v == -65.0
        assert n.eta == 0.0

    def test_step_returns_binary(self):
        assert GIFPopulationNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = GIFPopulationNeuron()
        assert sum(n.step(5.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = GIFPopulationNeuron()
        assert sum(n.step(30.0) for _ in range(10000)) > 20

    def test_stochastic(self):
        """Escape-rate model is stochastic — two runs should differ."""
        n1 = GIFPopulationNeuron()
        n2 = GIFPopulationNeuron()
        t1 = [n1.step(30.0) for _ in range(5000)]
        t2 = [n2.step(30.0) for _ in range(5000)]
        assert t1 != t2

    def test_adaptation_increases_after_spike(self):
        """eta should increase after spiking."""
        n = GIFPopulationNeuron()
        for _ in range(10000):
            if n.step(50.0):
                assert n.eta > 0
                break

    def test_adaptation_decays(self):
        """eta should decay toward zero without spikes."""
        n = GIFPopulationNeuron()
        n.eta = 10.0
        for _ in range(1000):
            n.step(0.0)
        assert n.eta < 1.0

    def test_rate_increases_with_input(self):
        n_low = GIFPopulationNeuron()
        n_high = GIFPopulationNeuron()
        s_low = sum(n_low.step(20.0) for _ in range(10000))
        s_high = sum(n_high.step(50.0) for _ in range(10000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 20.0, 50.0, 100.0]:
            n = GIFPopulationNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.eta), f"eta NaN at I={I}"

    def test_reset(self):
        n = GIFPopulationNeuron()
        for _ in range(5000):
            n.step(30.0)
        n.reset()
        assert n.v == -65.0
        assert n.eta == 0.0


class TestGIFNetwork:
    def test_population(self):
        assert Population(GIFPopulationNeuron, n=10, label="gif").n == 10

    def test_network_spikes(self):
        pop = Population(GIFPopulationNeuron, n=10, label="gif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestGIFAnalysis:
    def test_spike_count(self):
        n = GIFPopulationNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(30.0)
        assert spike_count(train) > 20
