# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCanonicalSign from former test_spike_stats_dimensionality.py

"""Focused suite: TestCanonicalSign from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403

class TestCanonicalSign:
    def test_empty(self) -> None:
        empty = np.empty((3, 0))
        npt.assert_array_equal(_DIM._canonical_sign(empty), empty)

    def test_flips_negative_dominant_column(self) -> None:
        comps = np.array([[-0.9, 0.1], [0.2, 0.8]])
        fixed = _DIM._canonical_sign(comps)
        # column 0's dominant entry (-0.9) becomes positive; column 1 unchanged
        assert fixed[0, 0] > 0
        npt.assert_allclose(fixed[:, 1], comps[:, 1])

    def test_zero_column_keeps_sign(self) -> None:
        comps = np.array([[0.0, 0.5], [0.0, -0.5]])
        fixed = _DIM._canonical_sign(comps)
        npt.assert_array_equal(fixed[:, 0], comps[:, 0])
