# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: InhomogeneousPoissonNeuron

"""Full pipeline test for InhomogeneousPoissonNeuron (Cox 1955).

Doubly stochastic Poisson: P(spike) = rate_hz · dt / 1000."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.inhomogeneous_poisson import InhomogeneousPoissonNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestIPoissonIsolation:
    def test_construction(self):
        n = InhomogeneousPoissonNeuron()
        assert n.dt_ms == 1.0

    def test_step_returns_binary(self):
        assert InhomogeneousPoissonNeuron().step(0.0) in (0, 1)

    def test_zero_rate_no_spikes(self):
        n = InhomogeneousPoissonNeuron()
        assert sum(n.step(0.0) for _ in range(10000)) == 0

    def test_negative_rate_no_spikes(self):
        n = InhomogeneousPoissonNeuron()
        assert sum(n.step(-50.0) for _ in range(10000)) == 0

    def test_spikes_at_rate(self):
        """100 Hz → ~100 spikes per 1000 ms (1000 steps)."""
        n = InhomogeneousPoissonNeuron()
        s = sum(n.step(100.0) for _ in range(10000))
        assert 500 < s < 1500

    def test_rate_proportional(self):
        n_low = InhomogeneousPoissonNeuron()
        n_high = InhomogeneousPoissonNeuron()
        s_low = sum(n_low.step(20.0) for _ in range(10000))
        s_high = sum(n_high.step(200.0) for _ in range(10000))
        assert s_high > s_low * 3

    def test_time_varying_rate(self):
        """Alternating high/low rate should produce uneven spike train."""
        n = InhomogeneousPoissonNeuron()
        spikes_high = sum(n.step(500.0) for _ in range(5000))
        spikes_low = sum(n.step(10.0) for _ in range(5000))
        assert spikes_high > spikes_low

    def test_stochastic(self):
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        t1 = [n1.step(100.0) for _ in range(1000)]
        t2 = [n2.step(100.0) for _ in range(1000)]
        assert t1 != t2

    def test_reset_noop(self):
        n = InhomogeneousPoissonNeuron()
        n.reset()
        assert n.dt_ms == 1.0

    def test_custom_dt(self):
        n = InhomogeneousPoissonNeuron(dt_ms=0.1)
        s = sum(n.step(1000.0) for _ in range(10000))
        assert 500 < s < 1500


class TestIPoissonNetwork:
    def test_population(self):
        assert Population(InhomogeneousPoissonNeuron, n=10, label="ip").n == 10


class TestIPoissonAnalysis:
    def test_spike_count(self):
        n = InhomogeneousPoissonNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(100.0)
        assert 500 < spike_count(train) < 1500
