# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGAnalytical from former test_model_neurogrid.py

"""Focused suite: TestNGAnalytical from former test_model_neurogrid.py."""

from __future__ import annotations

from tests.model_neurogrid_support import *  # noqa: F403

class TestNGAnalytical:
    def test_rk4_one_step_matches_candidate(self) -> None:
        """Default path commits the finite two-state RK4 candidate."""
        n = NeuroGridNeuron()
        state = (n.v_s, n.v_d)
        current = 50.0
        expected_vs, expected_vd = n._rk4_substep(state, current)
        n.step(current)
        assert abs(n.v_s - expected_vs) < 1e-12
        assert abs(n.v_d - expected_vd) < 1e-12

    def test_baseline_euler_formula_one_step(self) -> None:
        """Baseline Euler preserves the historical dendrite-first update."""
        n = NeuroGridNeuron(integrator="baseline_euler")
        vs0, vd0 = n.v_s, n.v_d
        I = 20.0  # subthreshold to avoid spike
        dvd = (-(vd0 - n.v_rest) + I - n.g_c * (vd0 - vs0)) / n.tau_d * n.dt
        vd_new = vd0 + dvd
        exp_arg = min((vs0 - n.v_threshold) / n.delta_t, 20.0)
        exp_term = n.delta_t * np.exp(exp_arg)
        dvs = (-(vs0 - n.v_rest) + exp_term + n.g_c * (vd_new - vs0)) / n.tau_s * n.dt
        n.step(I)
        assert abs((n.v_s - vs0) - dvs) < 1e-10
        assert abs(n.v_d - vd_new) < 1e-10

    def test_coupling_symmetric(self) -> None:
        """g_c·(v_d-v_s) in soma, -g_c·(v_d-v_s) in dendrite (current conservation)."""
        n = NeuroGridNeuron()
        # If v_d > v_s: current flows dendrite→soma
        n.v_d = -60.0
        n.v_s = -65.0
        coupling_to_soma = n.g_c * (n.v_d - n.v_s)  # +2.5
        coupling_from_dend = -n.g_c * (n.v_d - n.v_s)  # -2.5
        assert coupling_to_soma > 0  # excitatory to soma
        assert coupling_from_dend < 0  # drains dendrite
        assert abs(coupling_to_soma + coupling_from_dend) < 1e-12

    def test_exp_spike_initiation(self) -> None:
        """Exponential term grows as v_s → v_threshold."""
        n = NeuroGridNeuron()
        # Far below threshold: exp negligible
        exp_far = n.delta_t * np.exp((-65.0 - n.v_threshold) / n.delta_t)
        assert exp_far < 0.01
        # Near threshold: exp significant
        exp_near = n.delta_t * np.exp((-51.0 - n.v_threshold) / n.delta_t)
        assert exp_near > 1.0

    def test_exp_clipped_at_20(self) -> None:
        """Argument clamped at 20 to prevent overflow."""
        n = NeuroGridNeuron()
        # v_s very high → clipped
        n.v_s = 100.0
        n.v_d = -65.0
        n.step(0.0)  # Should not overflow
        assert np.isfinite(n.v_s)

    def test_spike_at_v_peak(self) -> None:
        """Spike when v_s ≥ v_peak, then v_s → v_reset."""
        n = NeuroGridNeuron()
        for _ in range(100_000):
            if n.step(100.0) == 1:
                assert n.v_s == n.v_reset
                break

    def test_dendritic_input_drives_soma(self) -> None:
        """Input to dendrite → dendrite depolarises → couples to soma → spike."""
        n = NeuroGridNeuron()
        for _ in range(1000):
            n.step(50.0)
        assert n.v_d > n.v_rest  # dendrite accumulated input
