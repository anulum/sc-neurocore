# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoid from former test_wilson_cowan_dynamics.py

"""Focused suite: TestSigmoid from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403

class TestSigmoid:
    """Published Wilson-Cowan 1972 two-term sigmoid:
        S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    Range is [-β, 1-β] with β = 1/(1+exp(aθ)). S(0) = 0 by construction."""

    def test_sigmoid_at_zero_is_zero(self):
        u = WilsonCowanUnit()
        assert abs(u._sigmoid(0.0)) < 1e-12

    def test_sigmoid_at_theta(self):
        """S(θ) = 1/2 − β."""
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        assert abs(u._sigmoid(u.theta) - (0.5 - baseline)) < 1e-12

    def test_sigmoid_monotone(self):
        u = WilsonCowanUnit()
        xs = np.linspace(-5, 15, 200)
        rs = np.array([u._sigmoid(x) for x in xs])
        assert (np.diff(rs) >= -1e-15).all()

    def test_sigmoid_asymptotes_respect_baseline(self):
        """S(x) → 1 − β as x → +∞, S(x) → −β as x → −∞.
        Input magnitude is limited to ~500 so the scalar `math.exp`
        does not overflow (math.exp raises OverflowError above ~709)."""
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        assert abs(u._sigmoid(500.0) - (1.0 - baseline)) < 1e-50
        assert abs(u._sigmoid(-500.0) - (-baseline)) < 1e-50

    def test_sigmoid_slope_at_theta(self):
        """dS/dx at θ equals a/4 (baseline subtraction does not shift the
        slope, only the level)."""
        u = WilsonCowanUnit()
        h = 1e-6
        slope = (u._sigmoid(u.theta + h) - u._sigmoid(u.theta - h)) / (2 * h)
        expected = u.a / 4.0
        assert abs(slope - expected) < 1e-4
