# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLMDynamics from former test_model_glm_neuron.py

"""Focused suite: TestGLMDynamics from former test_model_glm_neuron.py."""

from __future__ import annotations

from tests.model_glm_neuron_support import *  # noqa: F403

class TestGLMDynamics:
    def test_fires_with_strong_stimulus(self):
        n = GLMNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 10

    def test_silent_at_zero(self):
        """At μ=-3 and zero stimulus: very low rate."""
        n = GLMNeuron()
        spikes = _run(n, current=0.0, steps=1000)
        # May get 0-2 spikes (stochastic)
        assert len(spikes) < 50

    def test_rate_increases_with_stimulus(self):
        n_low = GLMNeuron()
        n_high = GLMNeuron()
        s_low = len(_run(n_low, current=2.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("stim", [0.0, 2.0, 5.0, 10.0])
    def test_stim_sweep(self, stim: float):
        n = GLMNeuron()
        spikes = [n.step(stim) for _ in range(1000)]

        assert set(spikes) <= {0, 1}
        np.testing.assert_array_equal(n._stim_buf, np.full(n.n_k, stim))
        assert set(n._spike_buf) <= {0.0, 1.0}
        assert np.all(np.isfinite(n._spike_buf))
