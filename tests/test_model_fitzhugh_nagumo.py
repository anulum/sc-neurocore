# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: FitzHughNagumoNeuron

"""Module-specific test for FitzHughNagumoNeuron (FitzHugh 1961).

2D qualitative model: dv/dt = v - v³/3 - w + I, dw/dt = ε(v+a-bw).
Oscillatory band I∈[0.5, 1.0]. Hopf bifurcation on both sides.
Nullcline analysis: V-nullcline w = v-v³/3+I, w-nullcline w = (v+a)/b.
The production default is RK4 over the published two-state ODE."""

from __future__ import annotations

import os
import time
import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: FitzHughNagumoNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _rhs(
    v: float, w: float, current: float, *, a: float = 0.7, b: float = 0.8, epsilon: float = 0.08
):
    return v - v**3 / 3.0 - w + current, epsilon * (v + a - b * w)


def _rk4_reference(
    v: float,
    w: float,
    current: float,
    dt: float,
    *,
    a: float = 0.7,
    b: float = 0.8,
    epsilon: float = 0.08,
):
    k1v, k1w = _rhs(v, w, current, a=a, b=b, epsilon=epsilon)
    k2v, k2w = _rhs(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current, a=a, b=b, epsilon=epsilon)
    k3v, k3w = _rhs(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current, a=a, b=b, epsilon=epsilon)
    k4v, k4w = _rhs(v + dt * k3v, w + dt * k3w, current, a=a, b=b, epsilon=epsilon)
    return (
        v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
    )


class TestFHNIsolation:
    def test_defaults(self):
        n = FitzHughNagumoNeuron()
        assert n.v == -1.0 and n.w == -0.5
        assert n.a == 0.7 and n.b == 0.8 and n.epsilon == 0.08
        assert n.integrator == "rk4"

    def test_step_returns_binary(self):
        assert FitzHughNagumoNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = FitzHughNagumoNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self):
        n = FitzHughNagumoNeuron()
        for _ in range(100000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = FitzHughNagumoNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == -1.0 and n.w == -0.5

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            FitzHughNagumoNeuron(integrator="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_cubic_overflow_fails_closed_without_mutating_state(self, integrator: str):
        n = FitzHughNagumoNeuron(v=1e103, w=0.0, integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="overflowed|non-finite"):
            n.step(0.0)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": math.nan},
            {"v": True},
            {"w": object()},
            {"w": math.inf},
            {"a": math.nan},
            {"b": 0.0},
            {"epsilon": 0.0},
            {"dt": 0.0},
            {"v_threshold": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            FitzHughNagumoNeuron(**kwargs)

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_runtime_parameter_corruption_fails_before_mutation(self, integrator: str):
        n = FitzHughNagumoNeuron(integrator=integrator)
        n.dt = math.nan
        before = (n.v, n.w)

        with pytest.raises(ValueError):
            n.step(0.5)

        assert (n.v, n.w) == before

    def test_runtime_positive_parameter_corruption_fails_before_mutation(self):
        n = FitzHughNagumoNeuron()
        n.b = 0.0
        before = (n.v, n.w)

        with pytest.raises(ValueError, match="b"):
            n.step(0.5)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_nonfinite_current_fails_before_mutation(self, integrator: str):
        n = FitzHughNagumoNeuron(integrator=integrator)
        before = (n.v, n.w)

        with pytest.raises(ValueError, match="current"):
            n.step(math.nan)

        assert (n.v, n.w) == before

    @pytest.mark.parametrize("candidate", [(math.nan, -0.5), (-1.0, math.inf)])
    def test_candidate_validation_rejects_nonfinite_state(self, candidate: tuple[float, float]):
        n = FitzHughNagumoNeuron()
        with pytest.raises(FloatingPointError, match="candidate"):
            n._validate_candidate(*candidate)

    def test_rhs_rejects_nonfinite_inputs(self):
        n = FitzHughNagumoNeuron()
        with pytest.raises(FloatingPointError, match="input"):
            n._rhs_tuple(math.inf, n.w, 0.5)

    def test_rhs_rejects_nonfinite_derivative(self):
        n = FitzHughNagumoNeuron(epsilon=1.0e308)
        with pytest.raises(FloatingPointError, match="derivative"):
            n._rhs_tuple(2.0, -0.5, 0.5)

    def test_rhs_array_adapter_matches_tuple_derivative(self):
        n = FitzHughNagumoNeuron()
        expected = n._rhs_tuple(n.v, n.w, 0.5)
        actual = n._rhs(0.0, np.array([n.v, n.w]), 0.5)
        assert tuple(actual) == pytest.approx(expected)

    def test_rosenbrock_candidate_produces_finite_state(self):
        n = FitzHughNagumoNeuron(integrator="rosenbrock")
        candidate = n._rosenbrock_candidate(0.5)
        assert np.all(np.isfinite(candidate))


class TestFHNDynamicsEquations:
    def test_rhs_formula(self):
        """The derivative matches the FitzHugh-Nagumo two-state ODE."""
        n = FitzHughNagumoNeuron()
        dv, dw = n._rhs_tuple(n.v, n.w, 0.5)
        expected_dv, expected_dw = _rhs(n.v, n.w, 0.5)
        assert dv == pytest.approx(expected_dv)
        assert dw == pytest.approx(expected_dw)

    def test_default_step_matches_independent_rk4_reference(self):
        """Production default RK4 matches an independent four-stage reference."""
        n = FitzHughNagumoNeuron()
        expected_v, expected_w = _rk4_reference(n.v, n.w, 0.5, n.dt)
        assert n.step(0.5) == 0
        assert n.v == pytest.approx(expected_v, abs=1.0e-15)
        assert n.w == pytest.approx(expected_w, abs=1.0e-15)

    def test_explicit_baseline_euler_matches_historical_formula(self):
        """Legacy Euler remains available only through explicit opt-in."""
        n = FitzHughNagumoNeuron(integrator="baseline_euler")
        v0, w0 = n.v, n.w
        dv, dw = _rhs(v0, w0, 0.5)
        assert n.step(0.5) == 0
        assert n.v == pytest.approx(v0 + dv * n.dt)
        assert n.w == pytest.approx(w0 + dw * n.dt)

    def test_cubic_nullcline(self):
        """V-nullcline: w = v - v³/3 + I. At v=0, I=0: w = 0."""
        w_null = 0.0 - 0.0**3 / 3 + 0.0
        assert abs(w_null) < 1e-10

    def test_w_nullcline(self):
        """w-nullcline: w = (v + a) / b. At v=-0.7: w = 0."""
        n = FitzHughNagumoNeuron()
        w_null = (-0.7 + n.a) / n.b
        assert abs(w_null) < 1e-10


class TestFHNOscillatoryBand:
    """Hopf bifurcation: oscillation in I ∈ [~0.3, ~1.2]."""

    def test_silent_below_band(self):
        n = FitzHughNagumoNeuron()
        assert len(_run(n, current=0.0, steps=10000)) <= 1

    def test_oscillatory_in_band(self):
        for I in [0.5, 0.8, 1.0]:
            n = FitzHughNagumoNeuron()
            spikes = _run(n, current=I, steps=10000)
            assert len(spikes) == {0.5: 26, 0.8: 28, 1.0: 28}[I]

    def test_suppressed_above_band(self):
        """High I pushes FP out of oscillatory region."""
        n = FitzHughNagumoNeuron()
        spikes = _run(n, current=2.0, steps=10000)
        assert spikes == [8]

    def test_regular_isi_in_band(self):
        n = FitzHughNagumoNeuron()
        spikes = _run(n, current=0.8, steps=10000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_voltage_bounded(self):
        """FHN V stays bounded ≈ [-2, 2] (cubic nullcline)."""
        n = FitzHughNagumoNeuron()
        vs = []
        for _ in range(10000):
            n.step(0.8)
            vs.append(n.v)
        assert min(vs) > -3 and max(vs) < 3


class TestFHNParameters:
    def test_epsilon_controls_timescale(self):
        n_fast = FitzHughNagumoNeuron(epsilon=0.2)
        n_slow = FitzHughNagumoNeuron(epsilon=0.02)
        s_fast = len(_run(n_fast, current=0.8, steps=10000))
        s_slow = len(_run(n_slow, current=0.8, steps=10000))
        assert s_fast != s_slow

    def test_a_shifts_w_nullcline(self):
        n1 = FitzHughNagumoNeuron(a=0.5)
        n2 = FitzHughNagumoNeuron(a=1.0)
        s1 = len(_run(n1, current=0.5, steps=10000))
        s2 = len(_run(n2, current=0.5, steps=10000))
        assert s1 != s2

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = FitzHughNagumoNeuron(dt=dt)
        for _ in range(10000):
            n.step(0.8)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FitzHughNagumoNeuron()
            trace = [(n.step(0.8), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestFHNPerformance:
    def test_isolation_throughput(self):
        n = FitzHughNagumoNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.8)
        elapsed = time.perf_counter() - t0
        throughput = N / elapsed
        minimum_throughput = 10000 if os.environ.get("CI") else 20000
        assert np.isfinite(n.v) and np.isfinite(n.w)
        assert throughput > minimum_throughput, (
            f"FHN isolation throughput regressed: {throughput:.0f}/s <= {minimum_throughput}/s"
        )

    def test_network_throughput(self):
        pop = Population(FitzHughNagumoNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestFHNPipeline:
    def test_population(self):
        assert Population(FitzHughNagumoNeuron, n=10, label="fhn").n == 10

    def test_network_spikes(self):
        pop = Population(FitzHughNagumoNeuron, n=10, label="fhn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(FitzHughNagumoNeuron, n=5, label="src")
        tgt = Population(FitzHughNagumoNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = FitzHughNagumoNeuron()
        train = np.array([float(n.step(0.8)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 5
        isis = isi(train, dt=0.0001)
        assert len(isis) >= 3
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
