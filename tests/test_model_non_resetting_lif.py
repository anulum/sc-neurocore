# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NonResettingLIFNeuron

"""Full pipeline test for NonResettingLIFNeuron (Kobayashi 2009 / Jolivet 2004).

LIF without voltage reset — only threshold rises after spike."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestNRLIFIsolation:
    def test_construction(self):
        n = NonResettingLIFNeuron()
        assert n.v == -65.0
        assert n.theta == -50.0

    def test_step_returns_binary(self):
        assert NonResettingLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = NonResettingLIFNeuron()
        assert sum(n.step(5.0) for _ in range(2000)) == 0

    def test_spikes_under_drive(self):
        n = NonResettingLIFNeuron()
        assert sum(n.step(20.0) for _ in range(5000)) > 5

    def test_no_voltage_reset(self):
        """V should NOT reset to V_rest after spike."""
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            if n.step(20.0):
                assert n.v >= n.theta - n.delta_theta
                break

    def test_theta_increases_on_spike(self):
        n = NonResettingLIFNeuron()
        theta0 = n.theta
        for _ in range(5000):
            if n.step(20.0):
                assert n.theta > theta0
                break

    def test_theta_decays(self):
        n = NonResettingLIFNeuron()
        n.theta = -30.0
        for _ in range(2000):
            n.step(0.0)
        assert n.theta < -30.0 + 1.0

    def test_numerical_stability(self):
        for I in [0.0, 10.0, 20.0, 50.0]:
            n = NonResettingLIFNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v)
            assert np.isfinite(n.theta)

    def test_reset(self):
        n = NonResettingLIFNeuron()
        for _ in range(2000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_rest

    def test_deterministic(self):
        n1 = NonResettingLIFNeuron()
        n2 = NonResettingLIFNeuron()
        for _ in range(500):
            assert n1.step(20.0) == n2.step(20.0)


class TestNRLIFNetwork:
    def test_population(self):
        assert Population(NonResettingLIFNeuron, n=10, label="nrlif").n == 10


class TestNRLIFAnalysis:
    def test_spike_count(self):
        n = NonResettingLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(20.0)
        assert spike_count(train) > 5
