# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: Loihi2Neuron

"""Full pipeline test for Loihi2Neuron (Intel 2021).

3-state-variable programmable neuron. All integer arithmetic.
s1=membrane, s2=synaptic, s3=adaptation. Bit-shift decays."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestLoihi2Isolation:
    def test_construction(self):
        n = Loihi2Neuron()
        assert n.s1 == 0
        assert n.s2 == 0
        assert n.s3 == 0

    def test_step_returns_binary(self):
        assert Loihi2Neuron().step(0) in (0, 1)

    def test_silent_at_zero(self):
        n = Loihi2Neuron()
        assert sum(n.step(0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = Loihi2Neuron()
        assert sum(n.step(100) for _ in range(1000)) > 100

    def test_three_state_variables(self):
        n = Loihi2Neuron()
        for _ in range(500):
            n.step(100)
        assert n.s1 != 0 or n.s2 != 0 or n.s3 != 0

    def test_adaptation_s3(self):
        """s3 should accumulate after spiking (s3_incr > 0)."""
        n = Loihi2Neuron()
        for _ in range(1000):
            if n.step(200):
                assert n.s3 > 0
                break

    def test_s3_decays(self):
        """s3 should decay without spikes."""
        n = Loihi2Neuron()
        n.s3 = 500
        for _ in range(500):
            n.step(0)
        assert n.s3 < 100

    def test_integer_arithmetic(self):
        n = Loihi2Neuron()
        for _ in range(100):
            n.step(100)
        assert isinstance(n.s1, int)
        assert isinstance(n.s2, int)
        assert isinstance(n.s3, int)

    def test_rate_increases_with_input(self):
        n_low = Loihi2Neuron()
        n_high = Loihi2Neuron()
        s_low = sum(n_low.step(50) for _ in range(1000))
        s_high = sum(n_high.step(200) for _ in range(1000))
        assert s_high > s_low

    def test_coupling_w12(self):
        """w12 controls s2→s1 coupling. w12=0 should reduce spiking."""
        n_on = Loihi2Neuron(w12=1)
        n_off = Loihi2Neuron(w12=0)
        s_on = sum(n_on.step(100) for _ in range(1000))
        s_off = sum(n_off.step(100) for _ in range(1000))
        assert s_on > s_off

    def test_numerical_stability(self):
        for I in [0, 50, 200, 1000]:
            n = Loihi2Neuron()
            for _ in range(2000):
                n.step(I)
            assert np.isfinite(n.s1)
            assert np.isfinite(n.s2)
            assert np.isfinite(n.s3)

    def test_reset(self):
        n = Loihi2Neuron()
        for _ in range(500):
            n.step(200)
        n.reset()
        assert n.s1 == 0
        assert n.s2 == 0
        assert n.s3 == 0

    def test_deterministic(self):
        n1 = Loihi2Neuron()
        n2 = Loihi2Neuron()
        for _ in range(200):
            assert n1.step(100) == n2.step(100)


class TestLoihi2Network:
    def test_population(self):
        assert Population(Loihi2Neuron, n=10, label="loihi2").n == 10


class TestLoihi2Analysis:
    def test_spike_count(self):
        n = Loihi2Neuron()
        train = np.zeros(1000, dtype=np.int8)
        for t in range(1000):
            train[t] = n.step(100)
        assert spike_count(train) > 100
