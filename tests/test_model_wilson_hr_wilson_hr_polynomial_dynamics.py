# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR polynomial dynamics

"""Verify the source polynomial ODE and continuous RK4 trajectory."""

from __future__ import annotations

from tests.model_wilson_hr_support import *


class TestWilsonHRPolynomialDynamics:
    def test_polynomial_formula(self) -> None:
        n = WilsonHRNeuron(v=-0.4, r=0.08)
        expected_dv, expected_dr = _rhs(n, n.v, n.r, 0.3)
        actual_dv, actual_dr = n._derivatives(n.v, n.r, 0.3)
        assert abs(actual_dv - expected_dv) < 1e-15
        assert abs(actual_dr - expected_dr) < 1e-15

    def test_step_matches_independent_rk4_reference(self) -> None:
        n = WilsonHRNeuron(v=-0.4, r=0.08)
        expected_v, expected_r = _rk4_reference(n, 0.3)
        assert n.step(0.3) == 0
        assert abs(n.v - expected_v) < 1e-15
        assert abs(n.r - expected_r) < 1e-15

    def test_upward_crossing_does_not_reset_rk4_candidate(self) -> None:
        n = WilsonHRNeuron(v=-0.01, r=0.05, v_peak=0.0, dt=0.02)
        previous_v = n.v
        candidate_v, candidate_r = _rk4_reference(n, 2.0)
        spike = n.step(2.0)
        assert spike == int(candidate_v >= n.v_peak and previous_v < n.v_peak)
        assert n.r == candidate_r
        assert n.v == candidate_v

    def test_r_nullcline(self) -> None:
        v = -0.7
        assert abs((1.35 * v + 1.03) - 0.085) < 1e-12

    def test_source_limit_cycle_stays_finite_without_reset(self) -> None:
        n = WilsonHRNeuron()
        vs = []
        for _ in range(50_000):
            n.step(0.3)
            vs.append(n.v)
        assert all(np.isfinite(vs))
        assert max(vs) > n.v_peak
