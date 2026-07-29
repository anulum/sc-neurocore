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
    def test_bump_stimulus_activates_field(self) -> None:
        """Gaussian bump input should create localised activation."""
        n = AmariNeuralField()
        x = np.arange(64)
        I_bump = np.exp(-0.5 * ((x - 32) / 5) ** 2) * 1.0
        for _ in range(500):
            n.step(I_bump)
        # Centre should be more active than edges
        state = amari_state(n)
        assert state[32] > state[0]

    def test_balanced_field_stays_bounded(self) -> None:
        """With balanced kernel, u should not diverge."""
        n = AmariNeuralField()
        I = np.ones(64) * 0.5
        for _ in range(1000):
            n.step(I)
        state = amari_state(n)
        assert np.all(np.isfinite(state))
        assert np.max(np.abs(state)) < 100

    def test_default_field_stays_bounded(self) -> None:
        """The corrected lateral-inhibition defaults remain finite and bounded."""
        n = AmariNeuralField()
        I = np.ones(64) * 1.0
        for _ in range(500):
            n.step(I)
        assert np.max(np.abs(amari_state(n))) < 100

    def test_mean_activation_returned(self) -> None:
        """step() returns the source-level active-site fraction."""
        n = AmariNeuralField()
        I = np.ones(64) * 0.5
        act = n.step(I)
        expected = float(np.count_nonzero(amari_state(n) > 0.0) / n.n)
        assert abs(act - expected) < 1e-10
