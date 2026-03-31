# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SigmoidRateNeuron

"""Full pipeline test for SigmoidRateNeuron (Wilson & Cowan style).

Continuous rate model: τ dr/dt = -r + σ(β(I-θ)). Returns float (rate),
not int spike. Network incompatible (float return)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.sigmoid_rate import SigmoidRateNeuron
from sc_neurocore.network.population import Population


class TestSigmoidRateIsolation:
    def test_defaults(self):
        n = SigmoidRateNeuron()
        assert n.r == 0.0 and n.tau == 10.0 and n.beta == 1.0 and n.theta == 0.0

    def test_step_returns_float(self):
        assert isinstance(SigmoidRateNeuron().step(0.0), (float, np.floating))

    def test_r_evolves(self):
        n = SigmoidRateNeuron()
        n.step(5.0)
        assert n.r > 0.0

    def test_state_finite(self):
        n = SigmoidRateNeuron()
        for _ in range(100000):
            n.step(5.0)
        assert np.isfinite(n.r)

    def test_reset(self):
        n = SigmoidRateNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.r == 0.0


class TestSigmoidRateTransferFunction:
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


class TestSigmoidRateParameters:
    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.5])
    def test_dt_stability(self, dt: float):
        n = SigmoidRateNeuron(dt=dt)
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.r)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SigmoidRateNeuron()
            trace = [n.step(3.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestSigmoidRatePerformance:
    def test_isolation_throughput(self):
        n = SigmoidRateNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(3.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000


class TestSigmoidRatePipeline:
    def test_population_creates(self):
        assert Population(SigmoidRateNeuron, n=10, label="sr").n == 10

    def test_returns_float_not_spike(self):
        """Rate model — returns float. Network.step_all limited."""
        n = SigmoidRateNeuron()
        assert isinstance(n.step(5.0), (float, np.floating))
