# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map dynamics contracts

"""Focused suite: TestRulkovMapDynamics from former test_model_rulkov_map.py."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


class TestRulkovMapDynamics:
    """Test the 3-branch piecewise map structure."""

    def test_branch1_x_le_0(self) -> None:
        """When x ≤ 0: x_new = alpha/(1-x) + y + I.

        At x=-1, y=-3, I=0: x_new = 4/(1-(-1)) + (-3) + 0 = 2 - 3 = -1.
        The map has a fixed point near here.
        """
        n = RulkovMapNeuron(x=-1.0, y=-3.0)
        n.step(0.0)
        # x_new = 4/2 + (-3) = -1.0 exactly (fixed point)
        assert abs(n.x - (-1.0)) < 1e-10

    def test_branch1_with_current_shifts(self) -> None:
        """Adding current shifts x_new upward."""
        n = RulkovMapNeuron(x=-1.0, y=-3.0)
        n.step(2.0)
        # x_new = 4/2 + (-3) + 2 = 1.0
        assert abs(n.x - 1.0) < 1e-10

    def test_branch3_reset(self) -> None:
        """When x ≥ alpha + y + I: x_new = -1.0 (hard reset)."""
        n = RulkovMapNeuron()
        # Force into branch 3: x > 0 and x >= alpha + y + I
        n.x = 5.0
        n.y = -3.0
        # alpha + y + 0 = 4 + (-3) = 1.0, x=5 >= 1.0 → branch 3
        n.step(0.0)
        assert n.x == -1.0

    def test_y_slow_variable_drift(self) -> None:
        """y evolves slowly (mu=0.001): y_new = y - μ(x+1) + μσ."""
        n = RulkovMapNeuron()
        y0 = n.y
        n.step(0.0)
        # At fixed point x=-1: dy = -μ(-1+1) + μσ = μσ = 0.001*(-1.6) = -0.0016
        dy = n.y - y0
        expected_dy = n.mu * n.sigma  # -0.0016
        assert abs(dy - expected_dy) < 1e-10

    def test_x_bounded(self) -> None:
        """x should stay bounded — map resets when x gets too large."""
        n = RulkovMapNeuron()
        xs = []
        for _ in range(10000):
            n.step(0.5)
            xs.append(n.x)
        x_values = np.array(xs)
        assert x_values.min() >= -3.0, f"x_min = {x_values.min():.3f}"
        assert x_values.max() < 10.0, f"x_max = {x_values.max():.3f}"
