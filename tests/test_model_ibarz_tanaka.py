# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: IbarzTanakaMapNeuron

"""Full pipeline test for IbarzTanakaMapNeuron (Ibarz et al. 2007).

Piecewise-linear bursting map: f(x) = α/(1-x) for x≤0, α+βx otherwise.
Slow y variable modulates bursting via µ."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestITMapIsolation:
    def test_construction(self):
        n = IbarzTanakaMapNeuron()
        assert n.x == -1.0
        assert n.y == -2.5

    def test_step_returns_binary(self):
        assert IbarzTanakaMapNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = IbarzTanakaMapNeuron()
        assert sum(n.step(0.5) for _ in range(5000)) == 0

    def test_spikes_under_drive(self):
        n = IbarzTanakaMapNeuron()
        assert sum(n.step(2.0) for _ in range(10000)) > 50

    def test_piecewise_f(self):
        """f(x) should switch at x=0."""
        n = IbarzTanakaMapNeuron()
        f_neg = n._f(-1.0)
        f_pos = n._f(1.0)
        assert abs(f_neg - 3.65 / 2.0) < 1e-10
        assert abs(f_pos - (3.65 + 0.25)) < 1e-10

    def test_slow_y_dynamics(self):
        """y changes slowly (µ=0.0005)."""
        n = IbarzTanakaMapNeuron()
        y0 = n.y
        for _ in range(1000):
            n.step(2.0)
        assert n.y != y0

    def test_reset_on_spike(self):
        """x should reset to x_reset when threshold crossed."""
        n = IbarzTanakaMapNeuron()
        for _ in range(10000):
            if n.step(2.0):
                assert n.x == n.x_reset
                break

    def test_rate_increases_with_input(self):
        n_low = IbarzTanakaMapNeuron()
        n_high = IbarzTanakaMapNeuron()
        s_low = sum(n_low.step(1.5) for _ in range(10000))
        s_high = sum(n_high.step(3.0) for _ in range(10000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 1.0, 2.0, 3.0]:
            n = IbarzTanakaMapNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.x), f"x NaN at I={I}"
            assert np.isfinite(n.y), f"y NaN at I={I}"

    def test_reset(self):
        n = IbarzTanakaMapNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.x == -1.0
        assert n.y == -2.5

    def test_deterministic(self):
        n1 = IbarzTanakaMapNeuron()
        n2 = IbarzTanakaMapNeuron()
        for _ in range(500):
            assert n1.step(2.0) == n2.step(2.0)


class TestITMapNetwork:
    def test_population(self):
        assert Population(IbarzTanakaMapNeuron, n=10, label="itm").n == 10


class TestITMapAnalysis:
    def test_spike_count(self):
        n = IbarzTanakaMapNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(2.0)
        assert spike_count(train) > 50
