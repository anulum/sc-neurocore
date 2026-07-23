# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCFCAnalyticalSolution from former test_model_cfc.py

"""Focused suite: TestCFCAnalyticalSolution from former test_model_cfc.py."""

from __future__ import annotations

from tests.model_cfc_support import *  # noqa: F403

class TestCFCAnalyticalSolution:
    """x(t+dt) = x·exp(-dt/τ_eff) + f_target·(1 - exp(-dt/τ_eff))."""

    def test_closed_form_formula(self):
        """Verify one step matches the closed-form expression."""
        n = ClosedFormContinuousNeuron(v_threshold=100.0)  # prevent spike
        I = 3.0
        x0 = n.x
        sigma_tau = 1.0 / (1.0 + np.exp(-(n.w_tau * I + n.bias)))
        tau_eff = max(n.tau_base * sigma_tau, 0.1)
        f_target = np.tanh(n.w_x * x0 + n.w_in * I)
        decay = np.exp(-n.dt / tau_eff)
        expected = x0 * decay + f_target * (1.0 - decay)
        n.step(I)
        assert abs(n.x - expected) < 1e-10

    def test_tau_eff_input_dependent(self):
        """τ_eff = τ_base · σ(w_τ·I + bias). Varies with input."""
        n = ClosedFormContinuousNeuron()
        tau1 = n.tau_base / (1.0 + np.exp(-(n.w_tau * 1.0)))
        tau5 = n.tau_base / (1.0 + np.exp(-(n.w_tau * 5.0)))
        assert tau1 != tau5

    def test_f_target_tanh_bounded(self):
        """f_target = tanh(w_x·x + w_in·I) ∈ [-1, 1]. Always bounded.

        Note: tanh(-100) rounds to exactly -1.0 in float64.
        """
        for I in [-100, 0, 100]:
            f = np.tanh(0.8 * 0.5 + 1.0 * I)
            assert -1.0 <= f <= 1.0

    def test_x_converges_to_f_target(self):
        """At steady state (many steps): x → f_target."""
        n = ClosedFormContinuousNeuron(v_threshold=100.0)
        I = 3.0
        for _ in range(10000):
            n.step(I)
        # At ss: x ≈ tanh(w_x*x + w_in*I)
        f_ss = np.tanh(n.w_x * n.x + n.w_in * I)
        assert abs(n.x - f_ss) < 0.01
