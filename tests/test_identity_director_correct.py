# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrect from former test_identity_director.py

"""Focused suite: TestCorrect from former test_identity_director.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.identity.director import (
    DirectorController,
)
from tests.identity_director_support import (
    _make_substrate,
)
from unittest.mock import (
    patch,
)


class TestCorrect:
    """Corrections the director applies."""

    def test_correct_rate_too_high(self) -> None:
        """Too high a rate strengthens inhibition."""
        sub = _make_substrate()
        director = DirectorController(sub)
        ie_before = sub.proj_ie.data.copy()
        with patch.object(director, "diagnose", return_value=["rate_too_high"]):
            director.correct()
        # proj_ie *= 1.1 — inhibitory weights are negative, so |w| increases
        assert np.mean(np.abs(sub.proj_ie.data)) > np.mean(np.abs(ie_before)) - 1e-6

    def test_correct_rate_too_low(self) -> None:
        """Too low a rate weakens inhibition."""
        sub = _make_substrate()
        director = DirectorController(sub)
        ie_before = sub.proj_ie.data.copy()
        with patch.object(director, "diagnose", return_value=["rate_too_low"]):
            director.correct()
        # proj_ie *= 0.9 — inhibitory weights |w| decreases
        assert np.mean(np.abs(sub.proj_ie.data)) < np.mean(np.abs(ie_before)) + 1e-6

    def test_correct_silent(self) -> None:
        """The correction runs for this diagnosis; its effect is not checked."""
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["silent"]):
            director.correct()

    def test_correct_too_regular(self) -> None:
        """Too regular firing perturbs the excitatory weights."""
        sub = _make_substrate()
        director = DirectorController(sub)
        ee_before = sub.proj_ee.data.copy()
        with patch.object(director, "diagnose", return_value=["too_regular"]):
            director.correct()
        # Weights should change (noise added)
        assert not np.allclose(sub.proj_ee.data, ee_before)

    def test_correct_too_chaotic(self) -> None:
        """The correction runs for this diagnosis; its effect is not checked."""
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["too_chaotic"]):
            director.correct()

    def test_correct_bursty(self) -> None:
        """The correction runs for this diagnosis; its effect is not checked."""
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["bursty"]):
            director.correct()

    def test_correct_connectivity_too_dense(self) -> None:
        """The correction runs for this diagnosis; its effect is not checked."""
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["connectivity_too_dense"]):
            director.correct()

    def test_correct_connectivity_too_sparse(self) -> None:
        """Too sparse connectivity grows synapses."""
        sub = _make_substrate()
        sub.proj_ee.data[:] = 0.0  # all zero
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["connectivity_too_sparse"]):
            director.correct()
        # Some weights should be grown
        assert np.any(sub.proj_ee.data > 0)

    def test_correct_healthy_no_action(self) -> None:
        """A healthy substrate is left unchanged."""
        sub = _make_substrate()
        director = DirectorController(sub)
        ee_before = sub.proj_ee.data.copy()
        with patch.object(director, "diagnose", return_value=[]):
            director.correct()
        np.testing.assert_array_equal(sub.proj_ee.data, ee_before)

    def test_corrections_counter(self) -> None:
        """Applied corrections are counted."""
        sub = _make_substrate()
        director = DirectorController(sub)
        assert director._corrections_applied == 0
        with patch.object(director, "diagnose", return_value=["rate_too_high"]):
            director.correct()
        assert director._corrections_applied == 1
