# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMParameters from former test_model_lnm.py

"""Focused suite: TestLNMParameters from former test_model_lnm.py."""

from __future__ import annotations

from tests.model_lnm_support import *  # noqa: F403

class TestLNMParameters:
    @pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
    def test_alpha_sweep(self, alpha: float):
        n = LearnableNeuronModel(alpha=alpha)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("gamma", [0.0, 0.05, 0.2])
    def test_gamma_sweep(self, gamma: float):
        n = LearnableNeuronModel(gamma=gamma)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)
