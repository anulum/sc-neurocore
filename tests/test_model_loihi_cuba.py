# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LoihiCUBANeuron

"""Full pipeline test for LoihiCUBANeuron (Davies 2018).

Intel Loihi CUBA LIF: 2-state integer (v, u). Division-based decay."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestLoihiCUBAIsolation:
    def test_construction(self):
        n = LoihiCUBANeuron()
        assert n.v == 0
        assert n.u == 0

    def test_step_returns_binary(self):
        assert LoihiCUBANeuron().step(0) in (0, 1)

    def test_silent_at_zero(self):
        n = LoihiCUBANeuron()
        assert sum(n.step(0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = LoihiCUBANeuron()
        assert sum(n.step(50) for _ in range(1000)) > 50

    def test_synaptic_current_u(self):
        """u should accumulate weighted input."""
        n = LoihiCUBANeuron()
        for _ in range(10):
            n.step(100)
        assert n.u > 0

    def test_u_decays(self):
        n = LoihiCUBANeuron()
        n.u = 500
        for _ in range(100):
            n.step(0)
        assert n.u < 50

    def test_integer_arithmetic(self):
        n = LoihiCUBANeuron()
        for _ in range(100):
            n.step(50)
        assert isinstance(n.v, int)
        assert isinstance(n.u, int)

    def test_rate_increases_with_input(self):
        n_low = LoihiCUBANeuron()
        n_high = LoihiCUBANeuron()
        s_low = sum(n_low.step(20) for _ in range(1000))
        s_high = sum(n_high.step(100) for _ in range(1000))
        assert s_high > s_low

    def test_reset(self):
        n = LoihiCUBANeuron()
        for _ in range(500):
            n.step(100)
        n.reset()
        assert n.v == 0
        assert n.u == 0

    def test_deterministic(self):
        n1 = LoihiCUBANeuron()
        n2 = LoihiCUBANeuron()
        for _ in range(200):
            assert n1.step(50) == n2.step(50)


class TestLoihiCUBANetwork:
    def test_population(self):
        assert Population(LoihiCUBANeuron, n=10, label="lcuba").n == 10


class TestLoihiCUBAAnalysis:
    def test_spike_count(self):
        n = LoihiCUBANeuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(50)
        assert spike_count(train) > 50
