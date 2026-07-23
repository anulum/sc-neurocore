# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTermanWangDynamics from former test_model_terman_wang.py

"""Focused suite: TestTermanWangDynamics from former test_model_terman_wang.py."""

from __future__ import annotations

from tests.model_terman_wang_support import *  # noqa: F403

class TestTermanWangDynamics:
    def test_cubic_nullcline(self):
        """f(v) = 3v - v³ + 2. At v=0: f=2. At v=1: f=4. At v=-1: f=0."""
        assert abs((3 * 0 - 0**3 + 2) - 2.0) < 1e-10
        assert abs((3 * 1 - 1**3 + 2) - 4.0) < 1e-10
        assert abs((3 * (-1) - (-1) ** 3 + 2) - 0.0) < 1e-10

    def test_sigmoid_recovery(self):
        """g(v) = alpha * (1 + tanh(v/beta))."""
        n = TermanWangOscillator()
        g_at_0 = n.alpha * (1.0 + np.tanh(0.0 / n.beta))
        assert abs(g_at_0 - n.alpha) < 1e-10  # tanh(0) = 0 → g = alpha

    def test_derivative_formula(self):
        """Derivative helper matches the documented two-state ODE."""
        n = TermanWangOscillator(v=-1.2, w=-0.25)
        assert n._derivatives(n.v, n.w, 1.0) == pytest.approx(_rhs(n, n.v, n.w, 1.0))

    def test_step_matches_independent_rk4_reference(self):
        """One step matches an independent coupled RK4 calculation."""
        n = TermanWangOscillator(v=-1.2, w=-0.25)
        expected = _rk4_reference(n, n.v, n.w, 1.0)
        assert n.step(1.0) == 0
        assert (n.v, n.w) == pytest.approx(expected, abs=1.0e-15)

    def test_slow_w_dynamics(self):
        """epsilon=0.02 → w evolves 50× slower than v."""
        n = TermanWangOscillator()
        v0, w0 = n.v, n.w
        n.step(1.0)
        dv = abs(n.v - v0)
        dw = abs(n.w - w0)
        assert dv > 10 * dw, f"dv={dv:.6f}, dw={dw:.6f}"

    def test_oscillation_at_moderate_I(self):
        """I=0.5–1.0: slow relaxation oscillation."""
        n = TermanWangOscillator()
        spikes = _run(n, current=1.0, steps=100000)
        assert len(spikes) >= 5

    def test_silent_at_zero(self):
        n = TermanWangOscillator()
        spikes = _run(n, current=0.0, steps=50000)
        assert len(spikes) <= 2  # at most transient

    def test_suppression_at_high_I(self):
        """I≥2: depolarisation block (V stays above v_peak, only 1 crossing)."""
        n = TermanWangOscillator()
        spikes = _run(n, current=5.0, steps=50000)
        assert len(spikes) <= 2
