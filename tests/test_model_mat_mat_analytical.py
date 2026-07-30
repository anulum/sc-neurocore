# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATAnalytical from former test_model_mat.py

"""Focused suite: TestMATAnalytical from former test_model_mat.py."""

from __future__ import annotations

from tests.model_mat_support import *


class TestMATAnalytical:
    def test_rk4_candidate_one_step(self):
        """The public step commits the candidate-first RK4 state."""
        n = SCResettingMATNeuron()
        I = 15.0
        expected_v, expected_theta1, expected_theta2 = n._rk4_candidate(I)
        n.step(I)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.theta1 - expected_theta1) < 1e-12
        assert abs(n.theta2 - expected_theta2) < 1e-12

    def test_rk4_separates_from_forward_euler(self):
        """Finite-dt MAT integration must not regress to raw forward Euler."""
        n = SCResettingMATNeuron(theta1=4.0, theta2=2.0, dt=2.0)
        I = 15.0
        euler_v = n.v + (-(n.v - n.v_rest) + n.resistance * I) / n.tau_m * n.dt
        expected_v, _, _ = n._rk4_candidate(I)
        assert abs(expected_v - euler_v) > 1e-3

    def test_theta1_exponential_decay(self):
        """RK4 threshold decay tracks the closed-form exponential."""
        n = SCResettingMATNeuron()
        n.theta1 = 5.0  # as if just spiked
        steps = 20
        for _ in range(steps):
            n.step(0.0)  # zero current to prevent new spikes
        expected = 5.0 * np.exp(-steps * n.dt / n.tau_1)
        assert abs(n.theta1 - expected) < 1e-3

    def test_theta2_exponential_decay(self):
        """RK4 slow-threshold decay tracks the closed-form exponential."""
        n = SCResettingMATNeuron()
        n.theta2 = 3.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        expected = 3.0 * np.exp(-steps * n.dt / n.tau_2)
        assert abs(n.theta2 - expected) < 1e-9

    def test_theta1_decays_faster_than_theta2(self):
        """tau_1 < tau_2 → theta1 decays faster."""
        n = SCResettingMATNeuron()
        n.theta1 = 10.0
        n.theta2 = 10.0
        for _ in range(50):
            n.step(0.0)
        assert n.theta1 < n.theta2

    def test_decay_ratio_matches_timescale(self):
        """After 1 tau: theta decays to ≈ 1/e of initial."""
        n = SCResettingMATNeuron()
        n.theta1 = 10.0
        for _ in range(int(n.tau_1 / n.dt)):
            n.step(0.0)
        expected = 10.0 / np.e
        assert abs(n.theta1 - expected) < 0.01

    def test_threshold_is_sum(self):
        """Effective threshold = V_base + theta1 + theta2."""
        n = SCResettingMATNeuron()
        n.theta1 = 3.0
        n.theta2 = 2.0
        threshold = n.v_threshold_base + n.theta1 + n.theta2
        assert abs(threshold - (-45.0)) < 1e-12

    def test_spike_increments_both_thetas(self):
        """On spike: theta1 += h1=5, theta2 += h2=3."""
        n = SCResettingMATNeuron()
        for _ in range(10_000):
            if n.step(30.0) == 1:
                # theta1 was decayed then incremented
                assert n.theta1 >= n.h1 * 0.9  # at least close to h1
                assert n.theta2 >= n.h2 * 0.9
                break

    def test_spike_retains_threshold_candidates(self):
        """Spike reset keeps the RK4-decayed threshold state before increments."""
        n = SCResettingMATNeuron()
        _, theta1_candidate, theta2_candidate = n._rk4_candidate(250.0)
        assert n.step(250.0) == 1
        assert n.v == n.v_reset
        assert abs(n.theta1 - (theta1_candidate + n.h1)) < 1e-12
        assert abs(n.theta2 - (theta2_candidate + n.h2)) < 1e-12

    def test_spike_resets_voltage(self):
        """On spike: V → V_reset."""
        n = SCResettingMATNeuron()
        for _ in range(10_000):
            if n.step(30.0) == 1:
                assert n.v == n.v_reset
                break

    def test_membrane_steady_state(self):
        """At steady state (no spike): V_ss = V_rest + R·I."""
        n = SCResettingMATNeuron()
        # Low current to avoid spiking
        I = 10.0
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.resistance * I
        # Should be close to steady state
        assert abs(n.v - expected_ss) < 1.0

    def test_invalid_current_preserves_state(self):
        """Invalid runtime current is rejected before mutating state."""
        n = SCResettingMATNeuron()
        before = (n.v, n.theta1, n.theta2)
        with pytest.raises(ValueError, match="input current"):
            n.step(float("nan"))
        assert (n.v, n.theta1, n.theta2) == before

    def test_invalid_state_preserves_state(self):
        """Corrupted threshold adaptation is rejected before mutation."""
        n = SCResettingMATNeuron()
        n.theta1 = -1.0
        before = (n.v, n.theta1, n.theta2)
        with pytest.raises(ValueError, match="threshold adaptation"):
            n.step(10.0)
        assert (n.v, n.theta1, n.theta2) == before
