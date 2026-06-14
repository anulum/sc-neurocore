# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: FitzHughRinzelNeuron

"""Module-specific tests for FitzHughRinzelNeuron.

The model is the FitzHugh-Nagumo fast subsystem plus ultra-slow Rinzel
modulation. Tests validate RK4 integration, three-timescale dynamics,
fail-closed numerical boundaries, pipeline wiring, and measured throughput.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron


def _run(neuron: FitzHughRinzelNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _rhs(
    v: float,
    w: float,
    y: float,
    current: float,
    *,
    a=0.7,
    b=0.8,
    c=-0.775,
    d=1.0,
    delta=0.08,
    mu=0.0001,
):
    return (
        v - v**3 / 3.0 - w + y + current,
        delta * (a + v - b * w),
        mu * (c - v - d * y),
    )


def _rk4_reference(v: float, w: float, y: float, current: float, dt: float):
    k1 = _rhs(v, w, y, current)
    k2 = _rhs(v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], y + 0.5 * dt * k1[2], current)
    k3 = _rhs(v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], y + 0.5 * dt * k2[2], current)
    k4 = _rhs(v + dt * k3[0], w + dt * k3[1], y + dt * k3[2], current)
    return (
        v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        y + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


class TestFHRIsolation:
    def test_defaults(self):
        n = FitzHughRinzelNeuron()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0
        assert n.delta == 0.08 and n.mu == 0.0001
        assert n.b == 0.8 and n.d == 1.0

    def test_step_returns_binary(self):
        assert FitzHughRinzelNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = FitzHughRinzelNeuron()
        initial = (n.v, n.w, n.y)
        for _ in range(1000):
            n.step(0.5)
        for name, v0, v1 in zip(["v", "w", "y"], initial, (n.v, n.w, n.y), strict=True):
            assert v0 != v1, f"{name} did not evolve"

    def test_state_finite(self):
        n = FitzHughRinzelNeuron()
        for _ in range(100_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_reset(self):
        n = FitzHughRinzelNeuron()
        for _ in range(500):
            n.step(0.5)
        n.reset()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0


class TestFHRThreeTimescales:
    def test_y_ultra_slow(self):
        """mu=0.0001 keeps y much slower than w over short horizons."""
        n = FitzHughRinzelNeuron()
        w0, y0 = n.w, n.y
        for _ in range(100):
            n.step(0.5)
        dw = abs(n.w - w0)
        dy = abs(n.y - y0)
        assert dw > 100 * dy, f"dw={dw:.6f}, dy={dy:.6f}"

    def test_y_modulates_oscillation(self):
        """Different y-nullcline offsets change the driven trajectory."""
        n1 = FitzHughRinzelNeuron(c=-0.5)
        n2 = FitzHughRinzelNeuron(c=-1.0)
        s1 = len(_run(n1, current=0.5, steps=10000))
        s2 = len(_run(n2, current=0.5, steps=10000))
        assert s1 != s2 or n1.y != pytest.approx(n2.y)


class TestFHRDynamics:
    def test_derivative_formula(self):
        """The derivative matches the three-state FitzHugh-Rinzel ODE."""
        n = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        assert n._derivatives(n.v, n.w, n.y, 0.5) == pytest.approx(_rhs(n.v, n.w, n.y, 0.5))

    def test_derivative_rejects_nonfinite_runtime_inputs(self):
        """The ODE primitive rejects corrupted nonfinite runtime values."""
        n = FitzHughRinzelNeuron()
        with pytest.raises(FloatingPointError, match="state and current must be finite"):
            n._derivatives(math.nan, n.w, n.y, 0.5)

    def test_step_matches_independent_rk4_reference(self):
        n = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        expected = _rk4_reference(n.v, n.w, n.y, 0.5, n.dt)
        assert n.step(0.5) == 0
        assert (n.v, n.w, n.y) == pytest.approx(expected, abs=1.0e-15)

    @pytest.mark.parametrize(
        "current, expected", [(0.0, 0), (0.5, 25), (0.8, 28), (1.0, 28), (2.0, 1)]
    )
    def test_deterministic_current_regimes(self, current: float, expected: int):
        n = FitzHughRinzelNeuron()
        assert len(_run(n, current=current, steps=10000)) == expected
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_v_bounded(self):
        n = FitzHughRinzelNeuron()
        vs = [n.v]
        for _ in range(10000):
            n.step(0.5)
            vs.append(n.v)
        assert min(vs) > -3 and max(vs) < 3

    def test_isi_regularity(self):
        n = FitzHughRinzelNeuron()
        spikes = _run(n, current=0.5, steps=10000)
        isis = np.diff(spikes[2:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.3


class TestFHRParameters:
    def test_mu_controls_y_speed(self):
        n_fast = FitzHughRinzelNeuron(mu=0.01)
        n_slow = FitzHughRinzelNeuron(mu=0.00001)
        for _ in range(5000):
            n_fast.step(0.5)
            n_slow.step(0.5)
        assert abs(n_fast.y) > abs(n_slow.y)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = FitzHughRinzelNeuron(dt=dt)
        for _ in range(10000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FitzHughRinzelNeuron()
            trace = [(n.step(0.5), n.v, n.w, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"v": math.nan}, "v.*finite"),
            ({"v": True}, "v.*finite"),
            ({"w": object()}, "w.*finite"),
            ({"b": 0.0}, "b.*positive"),
            ({"d": -1.0}, "d.*positive"),
            ({"delta": -0.1}, "delta.*positive"),
            ({"mu": 0.0}, "mu.*positive"),
            ({"dt": 0.0}, "dt.*positive"),
        ],
    )
    def test_rejects_invalid_numeric_configuration(self, kwargs: dict[str, float], match: str):
        with pytest.raises(ValueError, match=match):
            FitzHughRinzelNeuron(**kwargs)

    def test_rejects_nonfinite_current_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(ValueError, match="current"):
            neuron.step(float("nan"))

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_corrupted_runtime_parameter_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)
        neuron.mu = float("nan")

        with pytest.raises(ValueError, match="mu.*finite"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonpositive_runtime_parameter_without_mutation(self):
        neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)
        neuron.d = 0.0

        with pytest.raises(ValueError, match="d.*positive"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_overflow_candidate_without_mutation(self):
        # v = 1e155 makes the cube overflow to +inf; the exact `v*v*v` form
        # produces inf (rather than the libm-pow OverflowError) which the finite
        # guard rejects as a non-finite derivative — same contract, no mutation.
        neuron = FitzHughRinzelNeuron(v=1.0e155, w=0.2, y=0.1)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(FloatingPointError, match="derivative"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonfinite_derivative_without_mutation(self):
        neuron = FitzHughRinzelNeuron(mu=1.0e308)
        before = (neuron.v, neuron.w, neuron.y)

        with pytest.raises(FloatingPointError, match="derivative"):
            neuron.step(0.5)

        assert (neuron.v, neuron.w, neuron.y) == before

    def test_rejects_nonfinite_candidate_directly(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            FitzHughRinzelNeuron._validate_candidate(math.nan, -0.5, 0.0)


class TestFHRPerformance:
    def test_isolation_throughput(self):
        samples = []
        for _ in range(3):
            n = FitzHughRinzelNeuron()
            steps = 50_000
            t0 = time.perf_counter()
            for _ in range(steps):
                n.step(0.5)
            samples.append(time.perf_counter() - t0)

        best_seconds_per_step = min(samples) / steps
        assert best_seconds_per_step < 100e-6

    def test_network_throughput(self):
        pop = Population(FitzHughRinzelNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 3000


class TestFHRPipeline:
    def test_population(self):
        assert Population(FitzHughRinzelNeuron, n=10, label="fhr").n == 10

    def test_network_spikes(self):
        pop = Population(FitzHughRinzelNeuron, n=10, label="fhr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(FitzHughRinzelNeuron, n=5, label="src")
        tgt = Population(FitzHughRinzelNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = FitzHughRinzelNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 3
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
