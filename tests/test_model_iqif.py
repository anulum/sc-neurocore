# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: IntegerQIFNeuron

"""Full pipeline test for IntegerQIFNeuron (Lo et al. 2021).

Fixed-point QIF: V[t+1] = V + (V²>>k) + I, all integer arithmetic."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestIQIFIsolation:
    def test_construction(self):
        n = IntegerQIFNeuron()
        assert n.v == 0
        assert n.k == 6

    def test_step_returns_binary(self):
        assert IntegerQIFNeuron().step(0) in (0, 1)

    def test_silent_at_zero(self):
        n = IntegerQIFNeuron()
        assert sum(n.step(0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = IntegerQIFNeuron()
        assert sum(n.step(5) for _ in range(100)) > 50

    def test_integer_arithmetic(self):
        """All state should remain integer."""
        n = IntegerQIFNeuron()
        for _ in range(100):
            n.step(10)
        assert isinstance(n.v, int)

    def test_bit_shift(self):
        """V²>>k should produce the quadratic nonlinearity."""
        n = IntegerQIFNeuron()
        n.v = 32
        val = n.v * n.v >> n.k
        assert val == 16

    def test_v_min_clamp(self):
        """v should be clamped to v_min."""
        n = IntegerQIFNeuron()
        n.v = -3000
        n.step(0)
        assert n.v >= n.v_min

    def test_reset_on_spike(self):
        n = IntegerQIFNeuron()
        for _ in range(100):
            if n.step(10):
                assert n.v == n.v_reset
                break

    def test_rate_increases_with_input(self):
        n_low = IntegerQIFNeuron()
        n_high = IntegerQIFNeuron()
        s_low = sum(n_low.step(2) for _ in range(1000))
        s_high = sum(n_high.step(50) for _ in range(1000))
        assert s_high >= s_low

    def test_custom_k(self):
        """Larger k → more damped quadratic → fewer spikes at same I."""
        n_small = IntegerQIFNeuron(k=4)
        n_large = IntegerQIFNeuron(k=10)
        s_small = sum(n_small.step(3) for _ in range(1000))
        s_large = sum(n_large.step(3) for _ in range(1000))
        assert s_small >= s_large

    def test_reset(self):
        n = IntegerQIFNeuron()
        for _ in range(100):
            n.step(10)
        n.reset()
        assert n.v == 0

    def test_deterministic(self):
        n1 = IntegerQIFNeuron()
        n2 = IntegerQIFNeuron()
        for _ in range(200):
            assert n1.step(5) == n2.step(5)


class TestIQIFNetwork:
    def test_population(self):
        assert Population(IntegerQIFNeuron, n=10, label="iqif").n == 10


class TestIQIFAnalysis:
    def test_spike_count(self):
        n = IntegerQIFNeuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(5)
        assert spike_count(train) > 500
