# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRateTransferFunction from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRateTransferFunction from former test_model_sigmoid_rate.py."""

from __future__ import annotations

from tests.model_sigmoid_rate_support import *  # noqa: F403

class TestSigmoidRateTransferFunction:
    def test_single_step_matches_exact_relaxation(self):
        """The rate state follows the closed-form first-order relaxation."""
        n = SigmoidRateNeuron(r=0.25, tau=10.0, beta=2.0, theta=1.0, dt=0.5)
        current = 3.0
        sigma = _stable_sigmoid(n.beta, current, n.theta)
        expected = _exact_rate(n.r, sigma, n.dt, n.tau)
        assert n.step(current) == pytest.approx(expected, abs=1e-12)
        assert n.r == pytest.approx(expected, abs=1e-12)

    def test_large_timestep_exact_relaxation_remains_bounded(self):
        """Exact relaxation preserves the rate interval even when dt exceeds tau."""
        n = SigmoidRateNeuron(r=1.0, tau=0.1, dt=5.0)
        value = n.step(-100.0)
        expected = _exact_rate(1.0, _stable_sigmoid(n.beta, -100.0, n.theta), n.dt, n.tau)
        assert value == pytest.approx(expected, abs=1e-12)
        assert 0.0 <= n.r <= 1.0
        assert n.r < 1.0e-12

    def test_sigmoid_at_theta(self):
        """σ(β(I-θ)) at I=θ: σ(0) = 0.5."""
        n = SigmoidRateNeuron(theta=3.0)
        for _ in range(10000):
            n.step(3.0)
        assert abs(n.r - 0.5) < 0.01

    def test_sigmoid_monotonic(self):
        vals = []
        for I in [-5.0, 0.0, 1.0, 5.0, 10.0]:
            n = SigmoidRateNeuron()
            for _ in range(10000):
                n.step(I)
            vals.append(n.r)
        assert all(vals[j] <= vals[j + 1] + 0.01 for j in range(len(vals) - 1))

    def test_sigmoid_bounded_0_1(self):
        """Steady-state r ∈ [0, 1] (sigmoid output range)."""
        for I in [-100.0, 100.0]:
            n = SigmoidRateNeuron()
            for _ in range(10000):
                n.step(I)
            assert -0.01 <= n.r <= 1.01, f"I={I}: r={n.r}"

    def test_beta_sharpness(self):
        """Higher beta → sharper sigmoid → faster transition."""
        n_soft = SigmoidRateNeuron(beta=0.5, theta=0.0)
        n_sharp = SigmoidRateNeuron(beta=10.0, theta=0.0)
        for _ in range(10000):
            n_soft.step(1.0)
            n_sharp.step(1.0)
        # Sharp beta: r closer to 1.0
        assert n_sharp.r > n_soft.r

    def test_steady_state_exact(self):
        """r_ss = σ(β(I-θ))."""
        n = SigmoidRateNeuron(beta=2.0, theta=1.0)
        I = 3.0
        for _ in range(10000):
            n.step(I)
        expected = 1.0 / (1.0 + np.exp(-2.0 * (3.0 - 1.0)))
        assert abs(n.r - expected) < 0.01

    def test_tau_controls_convergence_speed(self):
        """Smaller tau → faster convergence to steady state."""
        n_fast = SigmoidRateNeuron(tau=1.0)
        n_slow = SigmoidRateNeuron(tau=100.0)
        for _ in range(100):
            n_fast.step(5.0)
            n_slow.step(5.0)
        # Fast should be closer to steady state
        assert n_fast.r > n_slow.r
