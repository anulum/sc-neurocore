# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ThresholdLinearRateNeuron

"""Full pipeline test for ThresholdLinearRateNeuron (Dayan & Abbott 2001).

Simplest rate model: r = gain · max(0, I - θ) (ReLU). Returns float.
Memoryless — r computed from current input, no state accumulation."""

from __future__ import annotations

import os
import time

import pytest

from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron
from sc_neurocore.network.population import Population


class TestThresholdLinearIsolation:
    def test_defaults(self):
        n = ThresholdLinearRateNeuron()
        assert n.r == 0.0 and n.theta == 0.0 and n.gain == 1.0

    def test_step_returns_float(self):
        assert isinstance(ThresholdLinearRateNeuron().step(1.0), float)

    def test_reset(self):
        n = ThresholdLinearRateNeuron()
        n.step(5.0)
        n.reset()
        assert n.r == 0.0


class TestThresholdLinearReLU:
    """Core: r = gain · max(0, I - θ). Pure ReLU."""

    def test_relu_below_threshold(self):
        """I < θ → r = 0."""
        n = ThresholdLinearRateNeuron(theta=2.0)
        r = n.step(1.0)
        assert r == 0.0

    def test_relu_at_threshold(self):
        """I = θ → r = 0."""
        n = ThresholdLinearRateNeuron(theta=2.0)
        r = n.step(2.0)
        assert r == 0.0

    def test_relu_above_threshold(self):
        """I > θ → r = gain · (I - θ)."""
        n = ThresholdLinearRateNeuron(theta=2.0, gain=1.0)
        r = n.step(5.0)
        assert abs(r - 3.0) < 1e-10

    def test_gain_scaling(self):
        """r = gain · max(0, I - θ). gain=2 → double output."""
        n = ThresholdLinearRateNeuron(theta=0.0, gain=2.0)
        r = n.step(3.0)
        assert abs(r - 6.0) < 1e-10

    def test_negative_input(self):
        """I < 0 (and θ=0) → r = 0 (ReLU clips negatives)."""
        n = ThresholdLinearRateNeuron(theta=0.0)
        r = n.step(-5.0)
        assert r == 0.0

    def test_linearity_above_threshold(self):
        """Above θ, output is linear in I."""
        n = ThresholdLinearRateNeuron(theta=1.0, gain=1.0)
        r2 = n.step(3.0)  # r = 2.0
        n.reset()
        r4 = n.step(5.0)  # r = 4.0
        assert abs(r4 / r2 - 2.0) < 1e-10

    def test_memoryless(self):
        """r depends only on current input, not history."""
        n = ThresholdLinearRateNeuron()
        n.step(10.0)
        r = n.step(0.0)
        assert r == 0.0  # no memory from previous input

    @pytest.mark.parametrize("theta", [-2.0, 0.0, 5.0, 10.0])
    def test_theta_shifts_activation(self, theta: float):
        n = ThresholdLinearRateNeuron(theta=theta)
        # At I = theta + 1: r = gain * 1 = 1.0
        r = n.step(theta + 1.0)
        assert abs(r - 1.0) < 1e-10


class TestThresholdLinearValidation:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
    def test_rejects_negative_or_non_finite_initial_rate(self, value: float):
        with pytest.raises(ValueError, match="r"):
            ThresholdLinearRateNeuron(r=value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_threshold(self, value: float):
        with pytest.raises(ValueError, match="theta"):
            ThresholdLinearRateNeuron(theta=value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
    def test_rejects_negative_or_non_finite_gain(self, value: float):
        with pytest.raises(ValueError, match="gain"):
            ThresholdLinearRateNeuron(gain=value)

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current_before_rate_mutation(self, current: float):
        n = ThresholdLinearRateNeuron(r=0.25)
        before = n.r
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.r == before

    def test_rejects_non_finite_runtime_rate_before_update(self):
        n = ThresholdLinearRateNeuron(r=0.25)
        n.r = float("nan")
        with pytest.raises(ValueError, match="runtime rate state"):
            n.step(1.0)
        assert n.r != n.r

    def test_rejects_non_finite_rate_output_before_mutation(self):
        n = ThresholdLinearRateNeuron(r=0.25, gain=1.0e308)
        before = n.r
        with pytest.raises(ValueError, match="rate output"):
            n.step(1.0e308)
        assert n.r == before


class TestThresholdLinearPerformance:
    def test_isolation_throughput(self):
        """ReLU is the fastest possible — no exp, no ODE."""
        n = ThresholdLinearRateNeuron()
        N = 500000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(3.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        minimum_rate = 400000 if os.environ.get("CI") else 500000
        assert n.r == 3.0
        assert rate > minimum_rate, f"isolation: {rate:.0f} steps/s, minimum={minimum_rate}"

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ThresholdLinearRateNeuron()
            trace = [n.step(float(x)) for x in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestThresholdLinearPipeline:
    def test_population_creates(self):
        assert Population(ThresholdLinearRateNeuron, n=10, label="relu").n == 10

    def test_returns_float_not_spike(self):
        n = ThresholdLinearRateNeuron()
        assert isinstance(n.step(5.0), float)
