# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: McKeanNeuron

"""Full pipeline test for McKeanNeuron (McKean 1970).

Piecewise-linear FHN caricature: 3-piece f(v), slow w recovery."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestMcKeanIsolation:
    def test_construction(self):
        n = McKeanNeuron()
        assert n.v == 0.0
        assert n.a == 0.25

    def test_step_returns_binary(self):
        assert McKeanNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = McKeanNeuron()
        assert sum(n.step(0.0) for _ in range(5000)) == 0

    def test_spikes_in_oscillatory_band(self):
        n = McKeanNeuron()
        assert sum(n.step(0.5) for _ in range(20000)) > 3

    def test_piecewise_f(self):
        """f(v) should have 3 pieces."""
        n = McKeanNeuron()
        assert n._f(0.0) == 0.0
        assert n._f(0.3) > 0
        assert n._f(0.8) < 0.5

    def test_w_recovery(self):
        n = McKeanNeuron()
        for _ in range(5000):
            n.step(0.5)
        assert n.w != 0.0

    def test_bounded_orbit(self):
        n = McKeanNeuron()
        for _ in range(20000):
            n.step(0.5)
        assert abs(n.v) < 5.0
        assert abs(n.w) < 5.0

    def test_numerical_stability(self):
        for I in [0.0, 0.3, 0.5, 1.0]:
            n = McKeanNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.v)
            assert np.isfinite(n.w)

    def test_reset(self):
        n = McKeanNeuron()
        for _ in range(5000):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0
        assert n.w == 0.0

    def test_deterministic(self):
        n1 = McKeanNeuron()
        n2 = McKeanNeuron()
        for _ in range(500):
            assert n1.step(0.5) == n2.step(0.5)


class TestMcKeanNetwork:
    def test_population(self):
        assert Population(McKeanNeuron, n=10, label="mck").n == 10


class TestMcKeanAnalysis:
    def test_spike_count(self):
        n = McKeanNeuron()
        train = np.zeros(20000, dtype=np.int8)
        for t in range(20000):
            train[t] = n.step(0.5)
        assert spike_count(train) > 3
