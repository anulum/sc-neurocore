# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GLMNeuron

"""Full pipeline test for GLMNeuron (Pillow et al. 2008).

Point-process GLM: lambda = exp(k·stim + h·spike_history + mu).
Stimulus filter k, post-spike filter h (refractoriness)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.glm_neuron import GLMNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestGLMIsolation:
    def test_construction(self):
        n = GLMNeuron()
        assert n.n_k == 10
        assert n.n_h == 20
        assert n.mu == -3.0

    def test_step_returns_binary(self):
        assert GLMNeuron().step(0.0) in (0, 1)

    def test_low_stim_few_spikes(self):
        n = GLMNeuron()
        s = sum(n.step(0.0) for _ in range(5000))
        assert s < 100

    def test_spikes_under_drive(self):
        n = GLMNeuron()
        assert sum(n.step(5.0) for _ in range(5000)) > 100

    def test_rate_increases_with_stim(self):
        n_low = GLMNeuron()
        n_high = GLMNeuron()
        s_low = sum(n_low.step(2.0) for _ in range(5000))
        s_high = sum(n_high.step(8.0) for _ in range(5000))
        assert s_high > s_low

    def test_stimulus_filter_shape(self):
        n = GLMNeuron()
        assert n.k.shape == (10,)
        assert n.k[0] > n.k[-1]

    def test_postspike_filter_shape(self):
        n = GLMNeuron()
        assert n.h.shape == (20,)
        assert n.h[0] < 0

    def test_postspike_refractoriness(self):
        """h filter is negative at short lags → suppresses immediate re-firing."""
        n = GLMNeuron()
        assert n.h[0] < -1.0

    def test_buffers_populated(self):
        n = GLMNeuron()
        for _ in range(50):
            n.step(5.0)
        assert np.any(n._stim_buf != 0)

    def test_numerical_stability(self):
        for stim in [0.0, 5.0, 10.0]:
            n = GLMNeuron()
            for _ in range(5000):
                n.step(stim)

    def test_reset(self):
        n = GLMNeuron()
        for _ in range(500):
            n.step(5.0)
        n.reset()
        assert np.all(n._stim_buf == 0)
        assert np.all(n._spike_buf == 0)

    def test_custom_filters(self):
        k = np.ones(5) * 0.3
        h = np.zeros(10)
        n = GLMNeuron(n_k=5, n_h=10, k=k, h=h)
        assert n.k.shape == (5,)
        for _ in range(500):
            n.step(5.0)


class TestGLMNetwork:
    def test_population(self):
        assert Population(GLMNeuron, n=10, label="glm").n == 10


class TestGLMAnalysis:
    def test_spike_count(self):
        n = GLMNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(5.0)
        assert spike_count(train) > 100
