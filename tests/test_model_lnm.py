# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LearnableNeuronModel

"""Full pipeline test for LearnableNeuronModel (Jahns et al. 2025).

V = alpha*V + beta*I + gamma*sigmoid(V). All params trainable."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.lnm import LearnableNeuronModel
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestLNMIsolation:
    def test_construction(self):
        n = LearnableNeuronModel()
        assert n.alpha == 0.9
        assert n.beta == 0.1
        assert n.gamma == 0.05

    def test_step_returns_binary(self):
        assert LearnableNeuronModel().step(0.0) in (0, 1)

    def test_silent_at_zero(self):
        n = LearnableNeuronModel()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = LearnableNeuronModel()
        assert sum(n.step(3.0) for _ in range(1000)) > 50

    def test_rate_increases_with_input(self):
        n_low = LearnableNeuronModel()
        n_high = LearnableNeuronModel()
        s_low = sum(n_low.step(1.0) for _ in range(1000))
        s_high = sum(n_high.step(5.0) for _ in range(1000))
        assert s_high > s_low

    def test_alpha_effect(self):
        """Lower alpha → faster decay → fewer spikes at weak drive."""
        n_fast = LearnableNeuronModel(alpha=0.5)
        n_slow = LearnableNeuronModel(alpha=0.99)
        s_fast = sum(n_fast.step(1.0) for _ in range(1000))
        s_slow = sum(n_slow.step(1.0) for _ in range(1000))
        assert s_slow > s_fast

    def test_beta_effect(self):
        """Higher beta → stronger input scaling."""
        n_low = LearnableNeuronModel(beta=0.05)
        n_high = LearnableNeuronModel(beta=0.5)
        s_low = sum(n_low.step(2.0) for _ in range(1000))
        s_high = sum(n_high.step(2.0) for _ in range(1000))
        assert s_high > s_low

    def test_gamma_nonlinearity(self):
        """gamma=0 removes sigmoid → pure linear dynamics."""
        n = LearnableNeuronModel(gamma=0.0)
        for _ in range(100):
            n.step(0.0)
        assert n.v == 0.0

    def test_numerical_stability(self):
        for I in [0.0, 3.0, 10.0]:
            n = LearnableNeuronModel()
            for _ in range(2000):
                n.step(I)
            assert np.isfinite(n.v)

    def test_reset(self):
        n = LearnableNeuronModel()
        for _ in range(200):
            n.step(3.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        n1 = LearnableNeuronModel()
        n2 = LearnableNeuronModel()
        for _ in range(200):
            assert n1.step(3.0) == n2.step(3.0)


class TestLNMNetwork:
    def test_population(self):
        assert Population(LearnableNeuronModel, n=10, label="lnm").n == 10

    def test_network_spikes(self):
        pop = Population(LearnableNeuronModel, n=10, label="lnm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0


class TestLNMAnalysis:
    def test_spike_count(self):
        n = LearnableNeuronModel()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(3.0)
        assert spike_count(train) > 50
