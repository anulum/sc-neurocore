# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLMParameters from former test_model_glm_neuron.py

"""Focused suite: TestGLMParameters from former test_model_glm_neuron.py."""

from __future__ import annotations

from tests.model_glm_neuron_support import *  # noqa: F403


class TestGLMParameters:
    @pytest.mark.parametrize("mu", [-5.0, -3.0, 0.0])
    def test_mu_sweep(self, mu: float):
        n = GLMNeuron(mu=mu)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("n_k", [5, 10, 20])
    def test_n_k_sweep(self, n_k: int):
        n = GLMNeuron(n_k=n_k)
        assert n.k.shape == (n_k,)
        for _ in range(500):
            n.step(5.0)

    @pytest.mark.parametrize("n_h", [10, 20, 40])
    def test_n_h_sweep(self, n_h: int):
        n = GLMNeuron(n_h=n_h)
        assert n.h.shape == (n_h,)
        for _ in range(500):
            n.step(5.0)
