# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanAnalytical from former test_model_mckean.py

"""Focused suite: TestMcKeanAnalytical from former test_model_mckean.py."""

from __future__ import annotations

from tests.model_mckean_support import *  # noqa: F403

class TestMcKeanAnalytical:
    def test_f_left_piece(self):
        n = McKeanNeuron()
        assert n._f(0.0) == 0.0
        assert abs(n._f(0.1) - (-0.1)) < 1e-12

    def test_f_middle_piece(self):
        n = McKeanNeuron()
        assert abs(n._f(0.125) - (0.125 - 0.25)) < 1e-12
        assert abs(n._f(0.4) - (0.4 - 0.25)) < 1e-12

    def test_f_right_piece(self):
        n = McKeanNeuron()
        assert abs(n._f(0.625) - (1.0 - 0.625)) < 1e-12
        assert abs(n._f(0.8) - 0.2) < 1e-12

    def test_f_continuity_at_breakpoints(self):
        n = McKeanNeuron()
        mid1, mid2 = n.a / 2.0, (1.0 + n.a) / 2.0
        assert abs(n._f(mid1 - 1e-10) - n._f(mid1)) < 1e-8
        assert abs(n._f(mid2 - 1e-10) - n._f(mid2)) < 1e-8

    def test_f_slopes(self):
        n = McKeanNeuron()
        eps = 1e-6
        slope_left = (n._f(0.05 + eps) - n._f(0.05 - eps)) / (2 * eps)
        slope_mid = (n._f(0.3 + eps) - n._f(0.3 - eps)) / (2 * eps)
        slope_right = (n._f(0.7 + eps) - n._f(0.7 - eps)) / (2 * eps)
        assert abs(slope_left - (-1.0)) < 0.01
        assert abs(slope_mid - 1.0) < 0.01
        assert abs(slope_right - (-1.0)) < 0.01

    def test_derivatives_match_mckean_rhs(self):
        n = McKeanNeuron(v=0.2, w=-0.1)
        dv, dw = n._derivatives(n.v, n.w, 0.5)
        expected_dv, expected_dw = _rhs(n, n.v, n.w, 0.5)
        assert abs(dv - expected_dv) < 1e-15
        assert abs(dw - expected_dw) < 1e-15

    def test_step_matches_independent_rk4_reference(self):
        n = McKeanNeuron(v=0.2, w=-0.1)
        expected_v, expected_w = _rk4_reference(n, 0.5)
        assert n.step(0.5) == 0
        assert abs(n.v - expected_v) < 1e-15
        assert abs(n.w - expected_w) < 1e-15

    def test_upward_threshold_crossing_reports_spike_once(self):
        n = McKeanNeuron(v=0.799, w=-0.2, dt=0.01)
        assert n.step(0.5) == 1
        assert n.v >= n.v_peak
        assert n.step(0.5) in (0, 1)

    def test_w_nullcline(self):
        n = McKeanNeuron()
        w_null = 0.0 / n.gamma
        assert abs(w_null) < 1e-12

    def test_v_nullcline(self):
        n = McKeanNeuron()
        i_ext = 0.5
        w_null = n._f(0.0) + i_ext
        assert abs(w_null - 0.5) < 1e-12
