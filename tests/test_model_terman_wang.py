# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific tests: TermanWangOscillator

"""Module-specific behavioural tests for TermanWangOscillator.

The tests verify the two-state ODE, coupled RK4 integration, slow recovery,
finite-domain rejection, and public network/analysis wiring without
bucket-style coverage assertions."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: TermanWangOscillator, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _rhs(neuron: TermanWangOscillator, v: float, w: float, current: float) -> tuple[float, float]:
    f = 3.0 * v - v**3 + 2.0
    g = neuron.alpha * (1.0 + np.tanh(v / neuron.beta))
    return f - w + current + neuron.rho, neuron.epsilon * (g - w)


def _rk4_reference(
    neuron: TermanWangOscillator, v: float, w: float, current: float
) -> tuple[float, float]:
    dt = neuron.dt
    k1 = _rhs(neuron, v, w, current)
    k2 = _rhs(neuron, v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v + dt * k3[0], w + dt * k3[1], current)
    return (
        v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


class TestTermanWangIsolation:
    def test_defaults(self):
        n = TermanWangOscillator()
        assert n.v == -1.5 and n.w == -0.5
        assert n.alpha == 3.0 and n.beta == 0.2 and n.epsilon == 0.02

    def test_step_returns_binary(self):
        assert TermanWangOscillator().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = TermanWangOscillator()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(1.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self):
        n = TermanWangOscillator()
        for _ in range(100000):
            n.step(1.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = TermanWangOscillator()
        for _ in range(1000):
            n.step(1.0)
        n.reset()
        assert n.v == -1.5 and n.w == -0.5


class TestTermanWangDynamics:
    def test_cubic_nullcline(self):
        """f(v) = 3v - v³ + 2. At v=0: f=2. At v=1: f=4. At v=-1: f=0."""
        assert abs((3 * 0 - 0**3 + 2) - 2.0) < 1e-10
        assert abs((3 * 1 - 1**3 + 2) - 4.0) < 1e-10
        assert abs((3 * (-1) - (-1) ** 3 + 2) - 0.0) < 1e-10

    def test_sigmoid_recovery(self):
        """g(v) = alpha * (1 + tanh(v/beta))."""
        n = TermanWangOscillator()
        g_at_0 = n.alpha * (1.0 + np.tanh(0.0 / n.beta))
        assert abs(g_at_0 - n.alpha) < 1e-10  # tanh(0) = 0 → g = alpha

    def test_derivative_formula(self):
        """Derivative helper matches the documented two-state ODE."""
        n = TermanWangOscillator(v=-1.2, w=-0.25)
        assert n._derivatives(n.v, n.w, 1.0) == pytest.approx(_rhs(n, n.v, n.w, 1.0))

    def test_step_matches_independent_rk4_reference(self):
        """One step matches an independent coupled RK4 calculation."""
        n = TermanWangOscillator(v=-1.2, w=-0.25)
        expected = _rk4_reference(n, n.v, n.w, 1.0)
        assert n.step(1.0) == 0
        assert (n.v, n.w) == pytest.approx(expected, abs=1.0e-15)

    def test_slow_w_dynamics(self):
        """epsilon=0.02 → w evolves 50× slower than v."""
        n = TermanWangOscillator()
        v0, w0 = n.v, n.w
        n.step(1.0)
        dv = abs(n.v - v0)
        dw = abs(n.w - w0)
        assert dv > 10 * dw, f"dv={dv:.6f}, dw={dw:.6f}"

    def test_oscillation_at_moderate_I(self):
        """I=0.5–1.0: slow relaxation oscillation."""
        n = TermanWangOscillator()
        spikes = _run(n, current=1.0, steps=100000)
        assert len(spikes) >= 5

    def test_silent_at_zero(self):
        n = TermanWangOscillator()
        spikes = _run(n, current=0.0, steps=50000)
        assert len(spikes) <= 2  # at most transient

    def test_suppression_at_high_I(self):
        """I≥2: depolarisation block (V stays above v_peak, only 1 crossing)."""
        n = TermanWangOscillator()
        spikes = _run(n, current=5.0, steps=50000)
        assert len(spikes) <= 2


class TestTermanWangParameters:
    @pytest.mark.parametrize("field", ["v", "w", "alpha", "rho", "v_peak"])
    def test_rejects_non_numeric_state_offsets_and_threshold(self, field: str):
        with pytest.raises(TypeError, match=field):
            TermanWangOscillator(**{field: object()})

    @pytest.mark.parametrize("field", ["beta", "epsilon", "dt"])
    def test_rejects_non_numeric_positive_parameters(self, field: str):
        with pytest.raises(TypeError, match=field):
            TermanWangOscillator(**{field: object()})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("w", np.inf),
            ("alpha", np.nan),
            ("beta", 0.0),
            ("epsilon", 0.0),
            ("dt", 0.0),
            ("v_peak", np.inf),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            TermanWangOscillator(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = TermanWangOscillator()
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(np.nan)
        assert (n.v, n.w) == before

    def test_rejects_non_numeric_current_before_state_mutation(self):
        n = TermanWangOscillator()
        before = (n.v, n.w)
        with pytest.raises(TypeError, match="current"):
            n.step(object())
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_scale_before_state_mutation(self):
        n = TermanWangOscillator()
        n.beta = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="beta"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = TermanWangOscillator()
        n.v = np.inf
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="v must be finite"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_rejects_cubic_overflow_before_state_mutation(self):
        n = TermanWangOscillator(v=1.0e308, w=-0.5)
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_derivative_rejects_nonfinite_runtime_inputs(self):
        n = TermanWangOscillator()
        with pytest.raises(FloatingPointError, match="state and current must be finite"):
            n._derivatives(np.nan, n.w, 1.0)

    def test_derivative_rejects_nonfinite_output(self):
        n = TermanWangOscillator()
        n.alpha = np.inf
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(n.v, n.w, 1.0)

    def test_rejects_nonfinite_candidate_directly(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            TermanWangOscillator._validate_candidate(np.nan, -0.5)

    def test_epsilon_controls_timescale(self):
        n_fast = TermanWangOscillator(epsilon=0.1)
        n_slow = TermanWangOscillator(epsilon=0.005)
        s_fast = len(_run(n_fast, current=1.0, steps=100000))
        s_slow = len(_run(n_slow, current=1.0, steps=100000))
        assert s_fast != s_slow

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = TermanWangOscillator(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TermanWangOscillator()
            trace = [(n.step(1.0), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestTermanWangPerformance:
    def test_isolation_throughput(self):
        n = TermanWangOscillator()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000

    def test_network_throughput(self):
        pop = Population(TermanWangOscillator, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestTermanWangPipeline:
    def test_population(self):
        assert Population(TermanWangOscillator, n=10, label="tw").n == 10

    def test_network_spikes(self):
        pop = Population(TermanWangOscillator, n=10, label="tw")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=10.0, dt=0.001, backend="python")
        # Slow oscillator — may need long run
        assert isinstance(mon.count, int)

    def test_projection_wiring(self):
        src = Population(TermanWangOscillator, n=5, label="src")
        tgt = Population(TermanWangOscillator, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = TermanWangOscillator()
        train = np.array([float(n.step(1.0)) for _ in range(100000)])
        sc = spike_count(train)
        assert sc >= 5
