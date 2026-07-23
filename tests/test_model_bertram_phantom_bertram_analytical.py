# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramAnalytical from former test_model_bertram_phantom.py

"""Focused suite: TestBertramAnalytical from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403

class TestBertramAnalytical:
    def test_boltzmann_midpoint(self):
        """At v=vh: σ(vh, vh, k) = 0.5 exactly."""
        n = BertramPhantomBurster()
        assert abs(n._boltz(-20.0, -20.0, 12.0) - 0.5) < 1e-12

    def test_boltzmann_limits(self):
        """σ → 1 for v >> vh, σ → 0 for v << vh."""
        n = BertramPhantomBurster()
        assert n._boltz(100.0, -20.0, 12.0) > 0.999
        assert n._boltz(-200.0, -20.0, 12.0) < 0.001

    def test_boltzmann_matches_reference(self):
        """Cross-check internal _boltz against reference implementation."""
        n = BertramPhantomBurster()
        for v in [-80, -50, -20, 0, 20]:
            for vh, k in [(n.v_m, n.s_m), (n.v_n, n.s_n), (n.v_s1, n.s_s1), (n.v_s2, n.s_s2)]:
                assert abs(n._boltz(v, vh, k) - _boltz(v, vh, k)) < 1e-14

    def test_derivative_formula_at_initial_state(self):
        """Derivative matches the published current-balance ODE."""
        n = BertramPhantomBurster()
        v0, s1_0, s2_0 = n.v, n.s1, n.s2
        I_ext = 200.0

        m_inf = _boltz(v0, n.v_m, n.s_m)
        n_inf = _boltz(v0, n.v_n, n.s_n)
        i_ca = n.g_ca * m_inf * (v0 - n.e_ca)
        i_k = n.g_k * n_inf * (v0 - n.e_k)
        i_s1 = n.g_s1 * s1_0 * (v0 - n.e_k)
        i_s2 = n.g_s2 * s2_0 * (v0 - n.e_k)
        i_l = n.g_l * (v0 - n.e_l)
        expected_dv_dt = (-i_ca - i_k - i_s1 - i_s2 - i_l + I_ext) / n.c_m

        actual_dv_dt, actual_ds1_dt, actual_ds2_dt = n._derivatives(v0, s1_0, s2_0, I_ext)

        assert abs(actual_dv_dt - expected_dv_dt) < 1e-12
        assert np.isfinite(actual_ds1_dt)
        assert np.isfinite(actual_ds2_dt)

    def test_ds1_derivative_formula(self):
        """ds1/dt = (s1_inf(V) - s1) / tau_s1."""
        n = BertramPhantomBurster()
        v0, s1_0 = n.v, n.s1
        s1_inf = _boltz(v0, n.v_s1, n.s_s1)
        expected_ds1_dt = (s1_inf - s1_0) / n.tau_s1
        _, actual_ds1_dt, _ = n._derivatives(v0, s1_0, n.s2, 0.0)
        assert abs(actual_ds1_dt - expected_ds1_dt) < 1e-14

    def test_ds2_derivative_formula(self):
        """ds2/dt = (s2_inf(V) - s2) / tau_s2."""
        n = BertramPhantomBurster()
        v0, s2_0 = n.v, n.s2
        s2_inf = _boltz(v0, n.v_s2, n.s_s2)
        expected_ds2_dt = (s2_inf - s2_0) / n.tau_s2
        _, _, actual_ds2_dt = n._derivatives(v0, n.s1, s2_0, 0.0)
        assert abs(actual_ds2_dt - expected_ds2_dt) < 1e-14

    def test_rk4_step_matches_independent_reference(self):
        """One production step matches an independent RK4 update of the three ODEs."""
        n = BertramPhantomBurster()
        state = np.array([n.v, n.s1, n.s2], dtype=float)
        current = 200.0

        def rhs(values):
            return np.array(n._derivatives(values[0], values[1], values[2], current))

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * n.dt * k1)
        k3 = rhs(state + 0.5 * n.dt * k2)
        k4 = rhs(state + n.dt * k3)
        expected = state + n.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        n.step(current)

        np.testing.assert_allclose([n.v, n.s1, n.s2], expected, rtol=0.0, atol=1e-12)

    def test_current_balance_at_rest(self):
        """Sum of ionic currents at initial state (v=-50, s1=0.1, s2=0.1)."""
        n = BertramPhantomBurster()
        v = n.v
        m_inf = _boltz(v, n.v_m, n.s_m)
        n_inf = _boltz(v, n.v_n, n.s_n)
        i_ca = n.g_ca * m_inf * (v - n.e_ca)
        i_k = n.g_k * n_inf * (v - n.e_k)
        i_s1 = n.g_s1 * n.s1 * (v - n.e_k)
        i_s2 = n.g_s2 * n.s2 * (v - n.e_k)
        i_l = n.g_l * (v - n.e_l)
        total = i_ca + i_k + i_s1 + i_s2 + i_l
        # Not zero at rest — model has non-trivial resting balance
        assert np.isfinite(total)
        # Ca is inward (negative I_Ca since v < e_ca), K is outward
        assert i_ca < 0  # v=-50 < e_ca=25 → (v - e_ca) < 0, inward
        assert i_k > 0  # v=-50 > e_k=-75 → (v - e_k) > 0, outward

    def test_five_ionic_currents_identified(self):
        """Model has 5 distinct currents: I_Ca, I_K, I_s1, I_s2, I_L."""
        n = BertramPhantomBurster()
        # Verify conductance parameters exist
        assert n.g_ca > 0 and n.g_k > 0 and n.g_s1 > 0
        assert n.g_s2 > 0 and n.g_l > 0
