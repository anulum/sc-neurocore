# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanSigmoid from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanSigmoid from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403


class TestWilsonCowanSigmoid:
    """Published two-term form, Wilson-Cowan 1972:
        S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    The subtracted baseline makes S(0) = 0 exactly. Range is therefore
    [-β, 1-β] where β = 1/(1+exp(aθ))."""

    def test_sigmoid_at_threshold(self):
        """S(θ) = 0.5 − β."""
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        assert abs(float(n._sigmoid(n.theta)) - (0.5 - baseline)) < 1e-12

    def test_sigmoid_at_zero(self):
        """S(0) = 0 by construction of the baseline subtraction."""
        n = WilsonCowanUnit()
        assert abs(float(n._sigmoid(0.0))) < 1e-12

    def test_sigmoid_monotonic(self):
        n = WilsonCowanUnit()
        vals = [float(n._sigmoid(x)) for x in [-5, 0, 4, 5, 10]]
        assert all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))

    def test_sigmoid_bounded_published_range(self):
        """Range is [−β, 1−β] where β = 1/(1+exp(aθ))."""
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        for x in [-50, -10, 0, 10, 50]:
            s = float(n._sigmoid(x))
            assert -baseline - 1e-12 <= s <= 1.0 - baseline + 1e-12
