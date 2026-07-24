# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLAnalytical from former test_model_morris_lecar.py

"""Focused suite: TestMLAnalytical from former test_model_morris_lecar.py."""

from __future__ import annotations

from tests.model_morris_lecar_support import *  # noqa: F403


class TestMLAnalytical:
    def test_m_inf_tanh(self):
        """m_∞(V) = 0.5·(1 + tanh((V-v1)/v2))."""
        n = MorrisLecarNeuron()
        for v in [-80, -60, -20, 0, 40]:
            expected = _m_inf(float(v), n.v1, n.v2)
            assert abs(n._m_inf(float(v)) - expected) < 1e-14

    def test_m_inf_range(self):
        """m_∞ ∈ (0, 1) — tanh bounded."""
        n = MorrisLecarNeuron()
        assert n._m_inf(-200.0) > 0
        assert n._m_inf(-200.0) < 0.01
        assert n._m_inf(200.0) > 0.99
        assert n._m_inf(200.0) < 1.0

    def test_m_inf_midpoint(self):
        """m_∞(v1) = 0.5."""
        n = MorrisLecarNeuron()
        assert abs(n._m_inf(n.v1) - 0.5) < 1e-12

    def test_w_inf_tanh(self):
        n = MorrisLecarNeuron()
        for v in [-80, -60, -20, 0, 40]:
            expected = _w_inf(float(v), n.v3, n.v4)
            assert abs(n._w_inf(float(v)) - expected) < 1e-14

    def test_w_inf_midpoint(self):
        """w_∞(v3) = 0.5."""
        n = MorrisLecarNeuron()
        assert abs(n._w_inf(n.v3) - 0.5) < 1e-12

    def test_lambda_positive(self):
        """λ(V) = φ·cosh(...) > 0 for all V (cosh > 0)."""
        n = MorrisLecarNeuron()
        for v in [-100, -60, 0, 50]:
            assert n._lam(float(v)) > 0

    def test_lambda_matches_reference(self):
        n = MorrisLecarNeuron()
        for v in [-80, -40, 0, 30]:
            expected = _lam(float(v), n.v3, n.v4, n.phi)
            assert abs(n._lam(float(v)) - expected) < 1e-14

    def test_lambda_minimum_at_v3(self):
        """λ minimum at V=v3 where cosh(0)=1 → λ_min = φ."""
        n = MorrisLecarNeuron()
        lam_v3 = n._lam(n.v3)
        assert abs(lam_v3 - n.phi) < 1e-12

    def test_dv_formula_one_step(self):
        """dV = (-I_Ca - I_K - I_L + I) / C_m · dt."""
        n = MorrisLecarNeuron(integrator="baseline_euler")
        v0, w0 = n.v, n.w
        I = 50.0
        m_inf = n._m_inf(v0)
        i_ca = n.g_ca * m_inf * (v0 - n.e_ca)
        i_k = n.g_k * w0 * (v0 - n.e_k)
        i_l = n.g_l * (v0 - n.e_l)
        expected_dv = (-i_ca - i_k - i_l + I) / n.c_m * n.dt
        n.step(I)
        assert abs((n.v - v0) - expected_dv) < 1e-10

    def test_dw_formula_one_step(self):
        """dw = λ(V)·(w_∞(V) - w) · dt."""
        n = MorrisLecarNeuron(integrator="baseline_euler")
        v0, w0 = n.v, n.w
        lam = n._lam(v0)
        w_inf = n._w_inf(v0)
        expected_dw = lam * (w_inf - w0) * n.dt
        n.step(0.0)
        assert abs((n.w - w0) - expected_dw) < 1e-14

    def test_default_rk4_separates_from_historical_euler(self):
        default = MorrisLecarNeuron(dt=0.1)
        baseline = MorrisLecarNeuron(dt=0.1, integrator="baseline_euler")
        current = 50.0

        default.step(current)
        baseline.step(current)

        assert default.integrator == "rk4"
        assert abs(default.v - baseline.v) > 1e-6
        assert abs(default.w - baseline.w) > 1e-8

    def test_current_balance_at_rest(self):
        """Three currents at initial state."""
        n = MorrisLecarNeuron()
        v = n.v
        m_inf = n._m_inf(v)
        i_ca = n.g_ca * m_inf * (v - n.e_ca)
        i_k = n.g_k * n.w * (v - n.e_k)
        i_l = n.g_l * (v - n.e_l)
        # I_Ca inward (v < e_ca), I_K outward (w=0 → negligible), I_L = 0 at v=e_l
        assert i_ca < 0  # inward
        assert abs(i_l) < 1e-10  # v=-60 = e_l
        assert abs(i_k) < 1e-10  # w=0
