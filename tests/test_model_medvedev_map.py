# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MedvedevMapNeuron

"""Full pipeline test for MedvedevMapNeuron (Medvedev 2005).

1D piecewise-monotone spiking map: x mod 1, chaotic dynamics."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMedvedevIsolation:
    def test_construction(self):
        n = MedvedevMapNeuron()
        assert n.x == 0.0
        assert n.alpha == 3.5

    def test_step_returns_binary(self):
        assert MedvedevMapNeuron().step(0.0) in (0, 1)

    def test_silent_at_zero(self):
        n = MedvedevMapNeuron()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_spikes_with_input(self):
        n = MedvedevMapNeuron()
        assert sum(n.step(0.2) for _ in range(5000)) > 100

    def test_x_bounded_mod1(self):
        """x is always mod 1 → in [0, 1)."""
        n = MedvedevMapNeuron()
        for _ in range(5000):
            n.step(0.3)
        assert 0.0 <= n.x < 1.0

    def test_piecewise_branches(self):
        """Below beta → scale, above → fold."""
        n = MedvedevMapNeuron()
        n.x = 0.1
        n.step(0.0)
        x_low = n.x
        n.x = 0.8
        n.step(0.0)
        x_high = n.x
        assert x_low != x_high

    def test_rate_increases_with_input(self):
        n_low = MedvedevMapNeuron()
        n_high = MedvedevMapNeuron()
        s_low = sum(n_low.step(0.1) for _ in range(5000))
        s_high = sum(n_high.step(0.5) for _ in range(5000))
        assert s_high > s_low

    def test_chaotic(self):
        """Tiny initial difference should amplify (sensitive dependence)."""
        n1 = MedvedevMapNeuron(x=0.1)
        n2 = MedvedevMapNeuron(x=0.1 + 1e-10)
        for _ in range(100):
            n1.step(0.2)
            n2.step(0.2)
        assert abs(n1.x - n2.x) > 1e-5

    def test_numerical_stability(self):
        for I in [0.0, 0.1, 0.3, 0.5]:
            n = MedvedevMapNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.x)

    def test_reset(self):
        n = MedvedevMapNeuron()
        for _ in range(500):
            n.step(0.2)
        n.reset()
        assert n.x == 0.0

    def test_deterministic(self):
        n1 = MedvedevMapNeuron()
        n2 = MedvedevMapNeuron()
        for _ in range(200):
            assert n1.step(0.2) == n2.step(0.2)


class TestMedvedevNetwork:
    def test_population(self):
        assert Population(MedvedevMapNeuron, n=10, label="med").n == 10


class TestMedvedevAnalysis:
    def test_spike_count(self):
        n = MedvedevMapNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(0.2)
        assert spike_count(train) > 100
