# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCNAnalytical from former test_model_courage_nekorkin_map.py

"""Focused suite: TestCNAnalytical from former test_model_courage_nekorkin_map.py."""

from __future__ import annotations

from tests.model_courage_nekorkin_map_support import *  # noqa: F403

class TestCNAnalytical:
    def test_breakpoints_formula(self):
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        exp_min, exp_max = _breakpoints()
        assert abs(jmin - exp_min) < 1e-15
        assert abs(jmax - exp_max) < 1e-15

    def test_f_lower_branch(self):
        """x <= Jmin: F(x) = -m0*x."""
        n = CourageNekorkinMapNeuron()
        jmin, _ = n._breakpoints()
        x = jmin - 0.05
        assert abs(n._f(x) - (-n.m0 * x)) < 1e-15

    def test_f_middle_branch(self):
        """Jmin < x < Jmax: F(x) = m1*(x - a)."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        x = 0.5 * (jmin + jmax)
        assert abs(n._f(x) - (n.m1 * (x - n.a))) < 1e-15

    def test_f_upper_branch(self):
        """x >= Jmax: F(x) = -m0*(x - 1)."""
        n = CourageNekorkinMapNeuron()
        _, jmax = n._breakpoints()
        x = jmax + 0.05
        assert abs(n._f(x) - (-n.m0 * (x - 1.0))) < 1e-15

    def test_f_continuous_at_breakpoints(self):
        """F is continuous at Jmin and Jmax by construction."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = n._breakpoints()
        d = 1e-9
        assert abs(n._f(jmin - d) - n._f(jmin + d)) < 1e-7
        assert abs(n._f(jmax - d) - n._f(jmax + d)) < 1e-7

    def test_x_update_formula_subthreshold(self):
        """Below d, H=0: x_new = x + F(x) - y + I."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y  # x0 = 0 < d -> H = 0
        cur = 0.05
        expected = x0 + n._f(x0) - y0 - n.beta * 0.0 + cur
        n.step(cur)
        assert abs(n.x - expected) < 1e-15

    def test_heaviside_active_above_d(self):
        """At x >= d, H=1 subtracts beta from the x update."""
        below = CourageNekorkinMapNeuron(x=0.30)  # >= d
        above = CourageNekorkinMapNeuron(x=0.30)
        x0 = 0.30
        below.beta = 0.0
        above.beta = 0.085
        below.step(0.0)
        above.step(0.0)
        # Difference between beta=0 and beta=0.085 updates is exactly -beta (H=1).
        assert abs((above.x - below.x) - (-0.085)) < 1e-15
        assert x0 >= above.d  # confirm H was active

    def test_y_update_formula(self):
        """y_new = y + eps*(x - J)."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y
        expected_dy = n.eps * (x0 - n.j)
        n.step(0.0)
        assert abs((n.y - y0) - expected_dy) < 1e-16

    def test_no_clip(self):
        """The canonical map has no clip — large states evolve by the raw map."""
        n = CourageNekorkinMapNeuron(x=5.0)
        x0 = 5.0
        expected = x0 + n._f(x0) - n.y - n.beta * 1.0
        n.step(0.0)
        assert abs(n.x - expected) < 1e-12
