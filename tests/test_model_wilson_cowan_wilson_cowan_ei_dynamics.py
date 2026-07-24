# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanEIDynamics from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanEIDynamics from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403


class TestWilsonCowanEIDynamics:
    """E/I population interaction — the core of Wilson-Cowan."""

    def test_e_increases_with_excitatory_input(self):
        """External input drives E upward."""
        n = WilsonCowanUnit()
        for _ in range(1000):
            n.step(10.0)
        assert n.e > 0.5

    def test_i_follows_e(self):
        """I is driven by E: w_ie·E enters the I sigmoid."""
        n = WilsonCowanUnit()
        for _ in range(1000):
            n.step(10.0)
        assert n.i > 0.1  # I has increased from following E

    def test_zero_input_low_activity(self):
        """Without input, E and I decay to low values."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(0.0)
        assert n.e < 0.05 and n.i < 0.05

    def test_e_bounded_0_1(self):
        """E rate should stay in [0, 1] (sigmoid output range)."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(10.0)
        assert 0.0 <= n.e <= 1.0

    def test_steady_state_at_high_input(self):
        """At high I_ext, E and I converge to steady state near 1.0."""
        n = WilsonCowanUnit()
        for _ in range(10000):
            n.step(10.0)
        e1 = n.e
        for _ in range(10000):
            n.step(10.0)
        assert abs(n.e - e1) < 0.001  # converged

    def test_step_uses_candidate_first_rk4_flow(self):
        n = WilsonCowanUnit(e=0.24, i=0.11, dt=0.35)
        expected_e, expected_i = _rk4_expected_state(n, 3.0)
        se = n._sigmoid(n.w_ee * n.e - n.w_ei * n.i + 3.0)
        si = n._sigmoid(n.w_ie * n.e - n.w_ii * n.i)
        euler_e = n.e + (-n.e + se) / n.tau_e * n.dt
        euler_i = n.i + (-n.i + si) / n.tau_i * n.dt

        result = n.step(3.0)

        assert result == pytest.approx(0.42143718680097664, abs=1e-15)
        assert n.e == pytest.approx(expected_e, abs=1e-15)
        assert n.i == pytest.approx(expected_i, abs=1e-15)
        assert abs(n.e - euler_e) > 1.0e-2
        assert abs(n.i - euler_i) > 1.0e-2

    def test_w_ee_controls_excitatory_recurrence(self):
        """Higher w_ee gives higher E→E feedback and higher E steady state."""
        n_low = WilsonCowanUnit(w_ee=5.0)
        n_high = WilsonCowanUnit(w_ee=15.0)
        for _ in range(10000):
            n_low.step(3.0)
            n_high.step(3.0)
        assert n_high.e > n_low.e

    def test_w_ei_controls_inhibition(self):
        """Higher w_ei gives higher I→E inhibition and lower E."""
        n_low = WilsonCowanUnit(w_ei=3.0)
        n_high = WilsonCowanUnit(w_ei=10.0)
        for _ in range(10000):
            n_low.step(5.0)
            n_high.step(5.0)
        assert n_low.e > n_high.e
