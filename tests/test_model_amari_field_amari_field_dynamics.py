# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariFieldDynamics from former test_model_amari_field.py

"""Focused suite: TestAmariFieldDynamics from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403


class TestAmariFieldDynamics:
    def test_bump_stimulus_activates_field(self):
        """Gaussian bump input should create localised activation."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        x = np.arange(64)
        I_bump = np.exp(-0.5 * ((x - 32) / 5) ** 2) * 1.0
        for _ in range(500):
            n.step(I_bump)
        # Centre should be more active than edges
        assert n.u[32] > n.u[0]

    def test_balanced_field_stays_bounded(self):
        """With balanced kernel, u should not diverge."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        for _ in range(1000):
            n.step(I)
        assert np.all(np.isfinite(n.u))
        assert np.max(np.abs(n.u)) < 100

    def test_default_params_diverge(self):
        """FINDING: default kernel sum > 1 → persistent input causes divergence."""
        n = AmariNeuralField()
        I = np.ones(64) * 1.0
        for _ in range(500):
            n.step(I)
        # u should have grown very large
        assert np.max(np.abs(n.u)) > 1000

    def test_mean_activation_returned(self):
        """step() returns mean of max(0, u) across field."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        act = n.step(I)
        expected = float(np.mean(np.maximum(n.u, 0.0)))
        assert abs(act - expected) < 1e-10
