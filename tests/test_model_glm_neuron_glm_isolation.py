# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLMIsolation from former test_model_glm_neuron.py

"""Focused suite: TestGLMIsolation from former test_model_glm_neuron.py."""

from __future__ import annotations

from tests.model_glm_neuron_support import *  # noqa: F403

class TestGLMIsolation:
    def test_defaults(self):
        n = GLMNeuron()
        assert n.n_k == 10 and n.n_h == 20
        assert n.mu == -3.0 and n.dt_ms == 1.0
        assert n.k.shape == (10,) and n.h.shape == (20,)

    def test_step_returns_binary(self):
        assert GLMNeuron().step(0.0) in (0, 1)

    def test_buffers_initialised_to_zero(self):
        n = GLMNeuron()
        assert np.all(n._stim_buf == 0.0)
        assert np.all(n._spike_buf == 0.0)

    def test_reset_clears_buffers(self):
        n = GLMNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert np.all(n._stim_buf == 0.0)
        assert np.all(n._spike_buf == 0.0)

    def test_stochastic_two_runs_differ(self):
        """Different RNG seeds → different spike trains."""
        n1 = GLMNeuron()
        n2 = GLMNeuron()
        t1 = [n1.step(5.0) for _ in range(1000)]
        t2 = [n2.step(5.0) for _ in range(1000)]
        assert t1 != t2
