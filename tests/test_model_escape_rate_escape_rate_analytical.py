# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateAnalytical from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateAnalytical from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403

class TestEscapeRateAnalytical:
    def test_v_steady_state(self):
        """V_ss = V_rest + R·I. Mean V should be near V_ss."""
        n = EscapeRateNeuron()
        for _ in range(10000):
            n.step(20.0)
        vs = []
        for _ in range(10000):
            n.step(20.0)
            vs.append(n.v)
        mean_v = np.mean(vs)
        v_ss = n.v_rest + n.resistance * 20.0
        assert abs(mean_v - v_ss) < 5.0

    def test_membrane_equation_one_step(self):
        """V_next = V_inf + (V - V_inf) * exp(-dt / tau_m)."""
        n = EscapeRateNeuron(seed=999)
        v0 = n.v
        I = 15.0
        v_inf = n.v_rest + n.resistance * I
        expected = v_inf + (v0 - v_inf) * math.exp(-n.dt / n.tau_m)
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-10

    def test_membrane_exact_flow_separates_from_forward_euler(self):
        n = EscapeRateNeuron(v=-65.0, dt=5.0, rho_0=1.0e-12, seed=999)
        v0 = n.v
        current = 10.0
        v_inf = n.v_rest + n.resistance * current
        euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau_m * n.dt
        expected = v_inf + (v0 - v_inf) * math.exp(-n.dt / n.tau_m)
        spike = n.step(current)
        assert spike == 0
        assert abs(n.v - expected) < 1e-10
        assert abs(n.v - euler) > 1e-3

    def test_rho0_scales_rate(self):
        n_low = EscapeRateNeuron(rho_0=0.0001)
        n_high = EscapeRateNeuron(rho_0=0.01)
        s_low = sum(n_low.step(30.0) for _ in range(50000))
        s_high = sum(n_high.step(30.0) for _ in range(50000))
        assert s_high > s_low

    def test_delta_u_controls_sensitivity(self):
        n_narrow = EscapeRateNeuron(delta_u=1.5)
        n_wide = EscapeRateNeuron(delta_u=6.0)
        s_narrow = sum(n_narrow.step(30.0) for _ in range(50000))
        s_wide = sum(n_wide.step(30.0) for _ in range(50000))
        assert s_narrow != s_wide
