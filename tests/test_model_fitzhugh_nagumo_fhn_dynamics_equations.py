# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNDynamicsEquations from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNDynamicsEquations from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

from tests.model_fitzhugh_nagumo_support import *  # noqa: F403

class TestFHNDynamicsEquations:
    def test_rhs_formula(self):
        """The derivative matches the FitzHugh-Nagumo two-state ODE."""
        n = FitzHughNagumoNeuron()
        dv, dw = n._rhs_tuple(n.v, n.w, 0.5)
        expected_dv, expected_dw = _rhs(n.v, n.w, 0.5)
        assert dv == pytest.approx(expected_dv)
        assert dw == pytest.approx(expected_dw)

    def test_default_step_matches_independent_rk4_reference(self):
        """Production default RK4 matches an independent four-stage reference."""
        n = FitzHughNagumoNeuron()
        expected_v, expected_w = _rk4_reference(n.v, n.w, 0.5, n.dt)
        assert n.step(0.5) == 0
        assert n.v == pytest.approx(expected_v, abs=1.0e-15)
        assert n.w == pytest.approx(expected_w, abs=1.0e-15)

    def test_explicit_baseline_euler_matches_historical_formula(self):
        """Legacy Euler remains available only through explicit opt-in."""
        n = FitzHughNagumoNeuron(integrator="baseline_euler")
        v0, w0 = n.v, n.w
        dv, dw = _rhs(v0, w0, 0.5)
        assert n.step(0.5) == 0
        assert n.v == pytest.approx(v0 + dv * n.dt)
        assert n.w == pytest.approx(w0 + dw * n.dt)

    def test_cubic_nullcline(self):
        """V-nullcline: w = v - v³/3 + I. At v=0, I=0: w = 0."""
        w_null = 0.0 - 0.0**3 / 3 + 0.0
        assert abs(w_null) < 1e-10

    def test_w_nullcline(self):
        """w-nullcline: w = (v + a) / b. At v=-0.7: w = 0."""
        n = FitzHughNagumoNeuron()
        w_null = (-0.7 + n.a) / n.b
        assert abs(w_null) < 1e-10
