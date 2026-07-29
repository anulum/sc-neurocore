# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariIsolation from former test_model_amari_field.py

"""Focused suite: TestAmariIsolation from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403


class TestAmariIsolation:
    def test_defaults(self) -> None:
        n = AmariNeuralField()
        assert n.n == 64 and n.tau == 10.0
        assert n.a_exc == 1.5 and n.b_inh == 0.75
        assert amari_state(n).shape == (64,)
        assert n._w.shape == (64,)

    def test_step_takes_array_returns_float(self) -> None:
        n = AmariNeuralField()
        result = n.step(np.zeros(64))
        assert isinstance(result, float)

    def test_step_broadcasts_scalar_field_level(self) -> None:
        n = AmariNeuralField(n=8)
        assert n.step(0.5) == 1.0

    def test_state_is_array(self) -> None:
        n = AmariNeuralField()
        assert isinstance(n.u, np.ndarray) and n.u.shape == (64,)

    def test_reset_zeros_field(self) -> None:
        n = AmariNeuralField()
        n.step(np.ones(64))
        n.reset()
        np.testing.assert_array_equal(n.u, 0.0)

    def test_zero_input_stays_zero(self) -> None:
        """Zero initial state + zero input → field stays at 0."""
        n = AmariNeuralField()
        for _ in range(500):
            n.step(np.zeros(64))
        assert np.allclose(amari_state(n), 0.0)
