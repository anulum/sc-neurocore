# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLMParameters from former test_model_glm_neuron.py

"""Focused suite: TestGLMParameters from former test_model_glm_neuron.py."""

from __future__ import annotations

import pytest
from sc_neurocore.neurons.models.glm_neuron import (
    GLMNeuron,
)
from sc_neurocore.neurons.seed_domain import NUMPY_SEED_DOMAIN
from tests.model_glm_neuron_support import (
    _run,
)


class TestGLMParameters:
    """Parameter behaviour of the GLMNeuron."""

    @pytest.mark.parametrize("seed", [NUMPY_SEED_DOMAIN[0] - 1, NUMPY_SEED_DOMAIN[1] + 1])
    def test_a_seed_outside_the_declared_domain_is_refused(self, seed: int) -> None:
        """A seed outside SEED_DOMAIN is refused at construction, with the domain named."""
        with pytest.raises(ValueError, match="the declared SEED_DOMAIN"):
            GLMNeuron(seed=seed)

    def test_the_domain_bounds_are_accepted(self) -> None:
        """Both ends of SEED_DOMAIN construct a model."""
        low, high = NUMPY_SEED_DOMAIN
        assert GLMNeuron(seed=low).seed == low
        assert GLMNeuron(seed=high).seed == high

    @pytest.mark.parametrize("mu", [-5.0, -3.0, 0.0])
    def test_mu_sweep(self, mu: float) -> None:
        """The model is exercised across a sweep of mu."""
        n = GLMNeuron(mu=mu)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("n_k", [5, 10, 20])
    def test_n_k_sweep(self, n_k: int) -> None:
        """The model is exercised across a sweep of n_k."""
        n = GLMNeuron(n_k=n_k)
        assert n.k.shape == (n_k,)
        for _ in range(500):
            n.step(5.0)

    @pytest.mark.parametrize("n_h", [10, 20, 40])
    def test_n_h_sweep(self, n_h: int) -> None:
        """The model is exercised across a sweep of n_h."""
        n = GLMNeuron(n_h=n_h)
        assert n.h.shape == (n_h,)
        for _ in range(500):
            n.step(5.0)
