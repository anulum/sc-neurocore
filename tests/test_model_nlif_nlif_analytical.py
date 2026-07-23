# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFAnalytical from former test_model_nlif.py

"""Focused suite: TestNLIFAnalytical from former test_model_nlif.py."""

from __future__ import annotations

from tests.model_nlif_support import *  # noqa: F403

class TestNLIFAnalytical:
    def test_cubic_term_at_rest(self):
        """At V=V_rest: a·(V_rest-V_rest)·(V_rest-V_crit) = 0."""
        n = NonlinearLIFNeuron()
        cubic = n.a * (n.v_rest - n.v_rest) * (n.v_rest - n.v_crit)
        assert abs(cubic) < 1e-14

    def test_cubic_term_above_v_crit(self):
        """V > V_crit and V > V_rest → positive feedback (runaway)."""
        n = NonlinearLIFNeuron()
        v = -35.0  # above both V_rest and V_crit
        cubic = n.a * (v - n.v_rest) * (v - n.v_crit)
        assert cubic > 0

    def test_cubic_term_between_rest_and_crit(self):
        """V_rest < V < V_crit → negative term (restoring)."""
        n = NonlinearLIFNeuron()
        v = -50.0
        cubic = n.a * (v - n.v_rest) * (v - n.v_crit)
        assert cubic < 0  # (positive) * (negative) = negative

    def test_rk4_candidate_one_step(self):
        """One step matches the candidate-first RK4 update."""
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        current = 5.0
        expected_v, expected_w = n._rk4_candidate(current)
        n.step(current)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.w - expected_w) < 1e-14
        assert n.v != v0
        assert n.w != w0

    def test_dw_formula_one_step(self):
        """dw = (b·(V-V_rest) - w) / tau_w · dt."""
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        expected_dw = (n.b * (v0 - n.v_rest) - w0) / n.tau_w * n.dt
        n.step(0.0)
        assert abs((n.w - w0) - expected_dw) < 1e-14

    def test_w_steady_state(self):
        """At steady state: w_ss = b·(V-V_rest)."""
        n = NonlinearLIFNeuron()
        # At rest: V=V_rest → w_ss = 0
        assert abs(n.b * (n.v_rest - n.v_rest)) < 1e-12

    def test_spike_resets_voltage(self):
        n = NonlinearLIFNeuron()
        for _ in range(10_000):
            if n.step(20.0) == 1:
                assert n.v == n.v_reset
                break

    def test_spike_threshold(self):
        """Spike on V ≥ V_threshold = -20."""
        n = NonlinearLIFNeuron()
        assert n.v_threshold == -20.0

    def test_v_nullcline(self):
        """V-nullcline: w = a·(V-V_rest)·(V-V_crit) + I."""
        n = NonlinearLIFNeuron()
        I = 10.0
        v = -50.0
        w_null = n.a * (v - n.v_rest) * (v - n.v_crit) + I
        assert np.isfinite(w_null)
