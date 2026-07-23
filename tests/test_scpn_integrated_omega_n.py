# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOmegaN from former test_scpn_integrated.py

"""Focused suite: TestOmegaN from former test_scpn_integrated.py."""

from __future__ import annotations

from tests.scpn_integrated_support import *  # noqa: F403

class TestOmegaN:
    def test_length(self):
        assert len(OMEGA_N) == 16

    def test_all_positive(self):
        assert np.all(OMEGA_N > 0)

    def test_l2_matches_gamma(self):
        """L2 neurochemical ≈ 40 Hz × 2π."""
        expected = 40.0 * 2 * np.pi
        np.testing.assert_allclose(OMEGA_N[1], expected, rtol=0.01)

    def test_l5_matches_one_hz(self):
        """L5 intentional frame ≈ 1 Hz × 2π."""
        expected = 1.0 * 2 * np.pi
        np.testing.assert_allclose(OMEGA_N[4], expected, rtol=0.01)
