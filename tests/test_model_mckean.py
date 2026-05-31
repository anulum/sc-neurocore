# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — module-specific behavioural test: McKeanNeuron

"""Behavioural contract for the McKean 1970 neuron surface.

The test surface is module-specific by default. Cross-module checks exercise the
real public workflow contract for using McKeanNeuron inside Population,
Projection, Network, SpikeMonitor, and spike-stat analysis APIs; they are not
coverage bucket tests.

Model equations:
dv/dt = f(v) - w + I
dw/dt = epsilon * (v - gamma*w)

f(v) = -v             if v < a/2
     = v - a          if a/2 <= v < (1+a)/2
     = 1 - v          if v >= (1+a)/2

The production integrator is candidate-first RK4 over the coupled (v, w) state.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, isi, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.mckean import McKeanNeuron


def _rhs(neuron: McKeanNeuron, v: float, w: float, current: float) -> tuple[float, float]:
    return neuron._f(v) - w + current, neuron.epsilon * (v - neuron.gamma * w)


def _rk4_reference(neuron: McKeanNeuron, current: float) -> tuple[float, float]:
    v0, w0 = neuron.v, neuron.w
    dt = neuron.dt
    k1 = _rhs(neuron, v0, w0, current)
    k2 = _rhs(neuron, v0 + 0.5 * dt * k1[0], w0 + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v0 + 0.5 * dt * k2[0], w0 + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v0 + dt * k3[0], w0 + dt * k3[1], current)
    return (
        v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _run(neuron: McKeanNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestMcKeanIsolation:
    def test_defaults(self):
        n = McKeanNeuron()
        assert n.v == 0.0 and n.w == 0.0
        assert n.a == 0.25 and n.epsilon == 0.01 and n.gamma == 0.5
        assert n.dt == 0.1 and n.v_peak == 0.8

    def test_step_returns_binary(self):
        assert McKeanNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self):
        n = McKeanNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0 and n.w != w0

    def test_state_finite_long_run(self):
        n = McKeanNeuron()
        for _ in range(100_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self):
        n = McKeanNeuron()
        for _ in range(5000):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0 and n.w == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = McKeanNeuron()
            trace = [(n.step(0.5), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_runtime_non_finite_state_fails_closed_without_mutating_w(self):
        n = McKeanNeuron()
        n.v = float("nan")
        before_w = n.w

        with pytest.raises(FloatingPointError, match="v must be finite"):
            n.step(0.5)

        assert np.isnan(n.v)
        assert n.w == before_w

    def test_runtime_update_overflow_fails_closed_without_mutating_state(self):
        n = McKeanNeuron(v=1e308, w=-1.7e308)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="finite"):
            n.step(1.7e308)

        assert (n.v, n.w) == before


class TestMcKeanAnalytical:
    def test_f_left_piece(self):
        n = McKeanNeuron()
        assert n._f(0.0) == 0.0
        assert abs(n._f(0.1) - (-0.1)) < 1e-12

    def test_f_middle_piece(self):
        n = McKeanNeuron()
        assert abs(n._f(0.125) - (0.125 - 0.25)) < 1e-12
        assert abs(n._f(0.4) - (0.4 - 0.25)) < 1e-12

    def test_f_right_piece(self):
        n = McKeanNeuron()
        assert abs(n._f(0.625) - (1.0 - 0.625)) < 1e-12
        assert abs(n._f(0.8) - 0.2) < 1e-12

    def test_f_continuity_at_breakpoints(self):
        n = McKeanNeuron()
        mid1, mid2 = n.a / 2.0, (1.0 + n.a) / 2.0
        assert abs(n._f(mid1 - 1e-10) - n._f(mid1)) < 1e-8
        assert abs(n._f(mid2 - 1e-10) - n._f(mid2)) < 1e-8

    def test_f_slopes(self):
        n = McKeanNeuron()
        eps = 1e-6
        slope_left = (n._f(0.05 + eps) - n._f(0.05 - eps)) / (2 * eps)
        slope_mid = (n._f(0.3 + eps) - n._f(0.3 - eps)) / (2 * eps)
        slope_right = (n._f(0.7 + eps) - n._f(0.7 - eps)) / (2 * eps)
        assert abs(slope_left - (-1.0)) < 0.01
        assert abs(slope_mid - 1.0) < 0.01
        assert abs(slope_right - (-1.0)) < 0.01

    def test_derivatives_match_mckean_rhs(self):
        n = McKeanNeuron(v=0.2, w=-0.1)
        dv, dw = n._derivatives(n.v, n.w, 0.5)
        expected_dv, expected_dw = _rhs(n, n.v, n.w, 0.5)
        assert abs(dv - expected_dv) < 1e-15
        assert abs(dw - expected_dw) < 1e-15

    def test_step_matches_independent_rk4_reference(self):
        n = McKeanNeuron(v=0.2, w=-0.1)
        expected_v, expected_w = _rk4_reference(n, 0.5)
        assert n.step(0.5) == 0
        assert abs(n.v - expected_v) < 1e-15
        assert abs(n.w - expected_w) < 1e-15

    def test_upward_threshold_crossing_reports_spike_once(self):
        n = McKeanNeuron(v=0.799, w=-0.2, dt=0.01)
        assert n.step(0.5) == 1
        assert n.v >= n.v_peak
        assert n.step(0.5) in (0, 1)

    def test_w_nullcline(self):
        n = McKeanNeuron()
        w_null = 0.0 / n.gamma
        assert abs(w_null) < 1e-12

    def test_v_nullcline(self):
        n = McKeanNeuron()
        i_ext = 0.5
        w_null = n._f(0.0) + i_ext
        assert abs(w_null - 0.5) < 1e-12


class TestMcKeanDynamics:
    def test_silent_at_zero_input(self):
        n = McKeanNeuron()
        assert len(_run(n, current=0.0, steps=20_000)) == 0

    def test_oscillatory_in_band(self):
        for current in [0.4, 0.5, 0.6]:
            n = McKeanNeuron()
            spikes = _run(n, current=current, steps=20_000)
            assert len(spikes) >= 3, f"I={current}: only {len(spikes)} spikes"

    def test_rate_monotonic(self):
        rates = []
        for current in [0.3, 0.5, 0.7]:
            n = McKeanNeuron()
            rates.append(len(_run(n, current=current, steps=20_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_fi_sweep(self, current: float):
        n = McKeanNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_regular_isi(self):
        n = McKeanNeuron()
        spikes = _run(n, current=0.5, steps=50_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_bounded_orbit(self):
        n = McKeanNeuron()
        vs, ws = [], []
        for _ in range(20_000):
            n.step(0.5)
            vs.append(n.v)
            ws.append(n.w)
        assert min(vs) > -2 and max(vs) < 2
        assert min(ws) > -2 and max(ws) < 2

    def test_upward_crossing_only(self):
        n = McKeanNeuron()
        prev_v = n.v
        for _ in range(20_000):
            spike = n.step(0.5)
            if spike == 1:
                assert prev_v < n.v_peak
            prev_v = n.v


class TestMcKeanParameters:
    @pytest.mark.parametrize("epsilon", [0.005, 0.01, 0.05])
    def test_epsilon_timescale(self, epsilon: float):
        n = McKeanNeuron(epsilon=epsilon)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    @pytest.mark.parametrize("a", [0.1, 0.25, 0.4])
    def test_a_breakpoint_sweep(self, a: float):
        n = McKeanNeuron(a=a)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("gamma", [0.3, 0.5, 0.8])
    def test_gamma_sweep(self, gamma: float):
        n = McKeanNeuron(gamma=gamma)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = McKeanNeuron(dt=dt)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)


class TestMcKeanPerformance:
    def test_isolation_throughput(self):
        n = McKeanNeuron()
        steps = 200_000
        t0 = time.perf_counter()
        for _ in range(steps):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        rate = steps / elapsed
        assert rate > 10_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(McKeanNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


class TestMcKeanPublicWorkflow:
    """Named workflow contract: McKean public surface inside the Python simulator."""

    def test_population(self):
        assert Population(McKeanNeuron, n=10, label="mck").n == 10

    def test_projection_wiring(self):
        src = Population(McKeanNeuron, n=5, label="src")
        tgt = Population(McKeanNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(McKeanNeuron, n=10, label="mck")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(20_000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self):
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(20_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self):
        n = McKeanNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1


class TestMcKeanValidation:
    @pytest.mark.parametrize("field", ["v", "w", "v_peak"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_threshold(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["v", "w", "v_peak"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_state_and_threshold(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("a", [0.0, -0.1, 1.0, 1.1, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_piecewise_breakpoint_parameter(self, a: float):
        with pytest.raises(ValueError, match="a"):
            McKeanNeuron(a=a)

    @pytest.mark.parametrize("value", [object(), "0.25", True])
    def test_rejects_non_numeric_piecewise_breakpoint_parameter(self, value: object):
        with pytest.raises(TypeError, match="a"):
            McKeanNeuron(a=value)

    @pytest.mark.parametrize("field", ["epsilon", "gamma", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["epsilon", "gamma", "dt"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_scales(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            McKeanNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = McKeanNeuron(v=0.25, w=-0.1)
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    @pytest.mark.parametrize("current", [object(), "0.5", True])
    def test_rejects_non_numeric_current_before_state_mutation(self, current: object):
        n = McKeanNeuron(v=0.25, w=-0.1)
        before = (n.v, n.w)
        with pytest.raises(TypeError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_scale_before_state_mutation(self):
        n = McKeanNeuron(v=0.25, w=-0.1)
        n.dt = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="dt"):
            n.step(0.5)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_breakpoint_before_state_mutation(self):
        n = McKeanNeuron(v=0.25, w=-0.1)
        n.a = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="a"):
            n.step(0.5)
        assert (n.v, n.w) == before

    def test_direct_derivative_rejects_non_finite_state(self):
        n = McKeanNeuron()
        with pytest.raises(FloatingPointError, match="state and current"):
            n._derivatives(np.nan, n.w, 0.5)

    def test_direct_derivative_rejects_non_finite_output(self):
        n = McKeanNeuron()
        n.epsilon = np.inf
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(0.2, -0.1, 0.5)

    def test_direct_candidate_validation_rejects_non_finite_candidate(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            McKeanNeuron._validate_candidate(np.nan, 0.0)
