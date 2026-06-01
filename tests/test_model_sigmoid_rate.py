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


def _stable_sigmoid(beta: float, current: float, theta: float) -> float:
    z = beta * (current - theta)
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    exp_z = np.exp(z)
    return exp_z / (1.0 + exp_z)


def _exact_rate(r: float, sigma: float, dt: float, tau: float) -> float:
    decay = np.exp(-dt / tau)
    return decay * r + (1.0 - decay) * sigma


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


class TestSigmoidRateValidation:
    @pytest.mark.parametrize("field", ["r", "beta", "theta"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_transfer_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SigmoidRateNeuron(**{field: value})

    @pytest.mark.parametrize("r", [-1.0e-12, 1.0 + 1.0e-12])
    def test_rejects_initial_rate_outside_unit_interval(self, r: float):
        with pytest.raises(ValueError, match="r must be in \\[0, 1\\]"):
            SigmoidRateNeuron(r=r)

    @pytest.mark.parametrize("field", ["tau", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_time_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SigmoidRateNeuron(**{field: value})

    def test_accepts_large_timestep_exact_relaxation(self):
        n = SigmoidRateNeuron(tau=0.1, dt=0.2)
        assert 0.0 <= n.step(1.0) <= 1.0

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_rate_mutation(self, current: float):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.r == before

    @pytest.mark.parametrize("field", ["r", "tau", "dt"])
    def test_rejects_corrupted_runtime_state_before_rate_mutation(self, field: str):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "r":
            assert n.r == before

    def test_rejects_runtime_rate_outside_unit_interval_before_mutation(self):
        n = SigmoidRateNeuron(r=0.25)
        n.r = 1.5
        with pytest.raises(ValueError, match="runtime rate state must be in \\[0, 1\\]"):
            n.step(1.0)
        assert n.r == 1.5

    @pytest.mark.parametrize("field", ["tau", "dt"])
    def test_rejects_non_positive_runtime_time_parameter_before_mutation(self, field: str):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        setattr(n, field, 0.0)
        with pytest.raises(ValueError, match="runtime time constants"):
            n.step(1.0)
        assert n.r == before

    def test_stable_sigmoid_rejects_nonsaturating_nan_argument(self):
        with pytest.raises(ValueError, match="sigmoid argument"):
            SigmoidRateNeuron._stable_sigmoid(np.inf, 1.0, 1.0)

    def test_large_runtime_timestep_preserves_rate_interval(self):
        n = SigmoidRateNeuron(r=1.0, tau=1.0e-308, dt=1.0e308)
        before = n.r
        assert n.step(-1.0e308) == pytest.approx(0.0, abs=1e-300)
        assert 0.0 <= n.r <= before

    def test_extreme_finite_drive_saturates_without_overflow_warning(self):
        n = SigmoidRateNeuron(beta=1.0e308, theta=0.0)
        with np.errstate(over="raise", invalid="raise"):
            high = n.step(1.0e308)
            n.reset()
            low = n.step(-1.0e308)
        assert 0.0 < high <= 1.0
        assert 0.0 <= low < high


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
