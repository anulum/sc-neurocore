# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFAnalytical from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFAnalytical from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403


class TestNRLIFAnalytical:
    def test_subthreshold_step_matches_exact_relaxation(self):
        """Linear membrane and threshold ODEs follow the closed-form solution."""
        n = NonResettingLIFNeuron(v=-60.0, theta=-40.0, dt=0.5)
        v0 = n.v
        theta0 = n.theta
        current = 4.0
        expected_v = _exact_relaxation(v0, n.v_rest + n.r_m * current, n.dt, n.tau_m)
        expected_theta = _exact_relaxation(theta0, n.theta_rest, n.dt, n.tau_theta)
        assert n.step(current) == 0
        assert n.v == pytest.approx(expected_v, abs=1e-12)
        assert n.theta == pytest.approx(expected_theta, abs=1e-12)

    def test_large_timestep_exact_relaxation_remains_bounded(self):
        """Exact relaxation stays inside the physical endpoint envelope for large dt."""
        n = NonResettingLIFNeuron(v=1000.0, theta=2000.0, dt=100.0)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(
            _exact_relaxation(1000.0, n.v_rest, n.dt, n.tau_m),
            abs=1e-12,
        )
        assert n.theta == pytest.approx(
            _exact_relaxation(2000.0, n.theta_rest, n.dt, n.tau_theta),
            abs=1e-12,
        )
        assert n.v_rest <= n.v <= 1000.0
        assert n.theta_rest <= n.theta <= 2000.0

    def test_dv_formula(self):
        """Voltage follows exact first-order relaxation toward V_rest + R·I."""
        n = NonResettingLIFNeuron()
        v0 = n.v
        I = 5.0
        expected_v = _exact_relaxation(v0, n.v_rest + n.r_m * I, n.dt, n.tau_m)
        n.step(I)
        assert abs(n.v - expected_v) < 1e-12

    def test_dtheta_formula(self):
        """Threshold follows exact first-order relaxation toward theta_rest."""
        n = NonResettingLIFNeuron()
        theta0 = n.theta
        expected_theta = _exact_relaxation(theta0, n.theta_rest, n.dt, n.tau_theta)
        n.step(0.0)  # subthreshold, no spike
        assert abs(n.theta - expected_theta) < 1e-14

    def test_no_voltage_reset_on_spike(self):
        """V does NOT reset after spike — key model feature."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            v_before = n.v
            if n.step(20.0) == 1:
                # V should be at or above where it was (not reset to V_rest)
                assert n.v >= n.v_rest
                # V was NOT set to v_rest or any v_reset
                assert n.v != n.v_rest or v_before == n.v_rest
                break

    def test_theta_increases_on_spike(self):
        """On spike: θ += Δθ."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            theta_before = n.theta
            if n.step(20.0) == 1:
                # theta increased by delta_theta (after decay within step)
                assert n.theta > theta_before
                break

    def test_theta_decays_toward_theta_rest(self):
        """θ decays exponentially toward θ_rest."""
        n = NonResettingLIFNeuron()
        n.theta = -30.0  # elevated
        for _ in range(5000):
            n.step(0.0)
        # Should decay toward theta_rest = -50
        assert n.theta < -30.0  # moved toward -50

    def test_theta_decay_rate(self):
        """After 1 tau_θ: θ decays by (1-1/e) of excess."""
        n = NonResettingLIFNeuron()
        n.theta = -40.0  # 10 above rest
        excess = n.theta - n.theta_rest  # = 10
        steps = int(n.tau_theta / n.dt)
        for _ in range(steps):
            n.step(0.0)
        remaining = n.theta - n.theta_rest
        # Should be ≈ excess/e ≈ 3.68
        expected = excess * np.exp(-1)
        assert abs(remaining - expected) < 0.5

    def test_membrane_steady_state(self):
        """At steady state (no spike): V_ss = V_rest + R·I."""
        n = NonResettingLIFNeuron()
        I = 5.0  # subthreshold
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.r_m * I
        assert abs(n.v - expected_ss) < 0.5

    def test_spike_condition(self):
        """Spike when V ≥ θ (dynamic threshold)."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            v_pre = n.v
            theta_pre = n.theta
            if n.step(20.0) == 1:
                # Before dv update within step, v crossed theta
                break
