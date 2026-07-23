# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLMAnalytical from former test_model_glm_neuron.py

"""Focused suite: TestGLMAnalytical from former test_model_glm_neuron.py."""

from __future__ import annotations

from tests.model_glm_neuron_support import *  # noqa: F403

class TestGLMAnalytical:
    def test_stimulus_filter_shape(self):
        """k = 0.5·exp(-arange(n_k)/3). Exponential decay."""
        n = GLMNeuron()
        expected = np.exp(-np.arange(10) / 3.0) * 0.5
        np.testing.assert_allclose(n.k, expected)

    def test_post_spike_filter_shape(self):
        """h = -5·exp(-t/2) + 0.5·exp(-t/10). Starts strongly negative."""
        n = GLMNeuron()
        t = np.arange(20)
        expected = -5.0 * np.exp(-t / 2.0) + 0.5 * np.exp(-t / 10.0)
        np.testing.assert_allclose(n.h, expected)

    def test_h_filter_refractoriness(self):
        """h[0] is strongly negative → suppresses firing after spike."""
        n = GLMNeuron()
        assert n.h[0] < -4.0  # -5 + 0.5 = -4.5

    def test_stimulus_buffer_circular(self):
        """New stimulus enters at index 0, old values shift right."""
        n = GLMNeuron(n_k=4, mu=-100.0)  # high mu to prevent spikes
        n.step(1.0)
        assert n._stim_buf[0] == 1.0
        n.step(2.0)
        assert n._stim_buf[0] == 2.0
        assert n._stim_buf[1] == 1.0

    def test_log_rate_clipping(self):
        """log_rate clipped to [-20, 20] → exp(20) ≈ 4.85e8."""
        n = GLMNeuron(mu=100.0)  # extreme mu
        spike = n.step(1000.0)

        assert spike == 1
        assert np.all(np.isfinite(n._stim_buf))
        assert np.all(np.isfinite(n._spike_buf))

    def test_baseline_rate_at_zero_input(self):
        """At zero stimulus, no history: log_rate = μ = -3.0 → λ = exp(-3) ≈ 0.05."""
        n = GLMNeuron()
        expected_lambda = np.exp(-3.0)
        expected_p = expected_lambda * 1.0 / 1000.0  # dt_ms=1, /1000
        # Very low probability per step
        assert expected_p < 0.001

    def test_spike_enters_spike_buffer(self):
        """After spike, spike_buf[0] = 1.0."""
        n = GLMNeuron(mu=10.0)  # high mu to guarantee spike

        assert n.step(10.0) == 1
        assert n._spike_buf[0] == 1.0
