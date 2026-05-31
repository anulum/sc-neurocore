# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — module-specific behavioural test: WilsonHRNeuron

"""Behavioural contract for the Wilson 1999 polynomial cortical neuron.

The module-specific tests validate the coupled Wilson-HR ODE, candidate-first
RK4 integration, finite-state error boundaries, reset semantics, and the named
public workflow contract inside the Python simulator.
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
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron


def _rhs(neuron: WilsonHRNeuron, v: float, r: float, current: float) -> tuple[float, float]:
    poly = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
    syn = -26.0 * r * (v + 0.92)
    return poly + syn + current, (-r + 1.35 * v + 1.03) / neuron.tau_r


def _rk4_reference(neuron: WilsonHRNeuron, current: float) -> tuple[float, float]:
    v0, r0 = neuron.v, neuron.r
    dt = neuron.dt
    k1 = _rhs(neuron, v0, r0, current)
    k2 = _rhs(neuron, v0 + 0.5 * dt * k1[0], r0 + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v0 + 0.5 * dt * k2[0], r0 + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v0 + dt * k3[0], r0 + dt * k3[1], current)
    return (
        v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        r0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _run(neuron: WilsonHRNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestWilsonHRIsolation:
    def test_defaults(self):
        n = WilsonHRNeuron()
        assert n.v == -0.7
        assert n.r == 0.1
        assert n.tau_r == 1.9
        assert n.v_peak == 0.4
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert WilsonHRNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = WilsonHRNeuron()
        v0, r0 = n.v, n.r
        for _ in range(100):
            n.step(0.3)
        assert n.v != v0 and n.r != r0

    def test_state_finite(self):
        n = WilsonHRNeuron()
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_reset(self):
        n = WilsonHRNeuron()
        for _ in range(100):
            n.step(0.3)
        n.reset()
        assert n.v == -0.7 and n.r == 0.1

    def test_spike_resets_v(self):
        n = WilsonHRNeuron()
        for _ in range(50_000):
            if n.step(0.3) == 1:
                assert n.v == -0.7
                break


class TestWilsonHRPolynomialDynamics:
    def test_polynomial_formula(self):
        n = WilsonHRNeuron(v=-0.4, r=0.08)
        expected_dv, expected_dr = _rhs(n, n.v, n.r, 0.3)
        actual_dv, actual_dr = n._derivatives(n.v, n.r, 0.3)
        assert abs(actual_dv - expected_dv) < 1e-15
        assert abs(actual_dr - expected_dr) < 1e-15

    def test_step_matches_independent_rk4_reference(self):
        n = WilsonHRNeuron(v=-0.4, r=0.08)
        expected_v, expected_r = _rk4_reference(n, 0.3)
        assert n.step(0.3) == 0
        assert abs(n.v - expected_v) < 1e-15
        assert abs(n.r - expected_r) < 1e-15

    def test_threshold_reset_after_rk4_candidate(self):
        n = WilsonHRNeuron(v=0.35, r=0.05, dt=0.02)
        candidate_v, candidate_r = _rk4_reference(n, 2.0)
        spike = n.step(2.0)
        assert spike == int(candidate_v >= n.v_peak)
        assert n.r == candidate_r
        if spike:
            assert n.v == -0.7

    def test_r_nullcline(self):
        v = -0.7
        assert abs((1.35 * v + 1.03) - 0.085) < 1e-12

    def test_v_bounded_by_reset(self):
        n = WilsonHRNeuron()
        vs = []
        for _ in range(50_000):
            n.step(0.3)
            vs.append(n.v)
        assert max(vs) <= n.v_peak + 0.1


class TestWilsonHRCurrentResponse:
    def test_low_current_regime_is_subthreshold(self):
        for current in [0.0, 0.3, 1.0]:
            n = WilsonHRNeuron()
            assert len(_run(n, current=current, steps=5_000)) == 0

    def test_moderate_current_regime_stays_finite(self):
        for current in [0.6, 0.8, 1.0]:
            n = WilsonHRNeuron()
            for _ in range(5_000):
                n.step(current)
            assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_drive_evokes_transient_spiking(self):
        n = WilsonHRNeuron()
        spikes = _run(n, current=2.0, steps=5_000)
        assert len(spikes) >= 1

    def test_high_drive_produces_more_transient_spikes_than_threshold_drive(self):
        n_low = WilsonHRNeuron()
        n = WilsonHRNeuron()
        low_spikes = _run(n_low, current=2.0, steps=5_000)
        high_spikes = _run(n, current=10.0, steps=5_000)
        assert len(high_spikes) > len(low_spikes)

    def test_fi_5_point_sweep(self):
        rates = {}
        for current in [0.0, 0.3, 0.6, 2.0, 10.0]:
            n = WilsonHRNeuron()
            rates[current] = len(_run(n, current=current, steps=5_000))
        assert rates[0.0] == rates[0.3] == rates[0.6] == 0
        assert rates[10.0] > rates[2.0] > rates[0.6]


class TestWilsonHRISI:
    def test_isi_variability_at_peak(self):
        n = WilsonHRNeuron()
        spikes = _run(n, current=0.3, steps=50_000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv > 0, f"CV(ISI) should be > 0, got {cv:.4f}"


class TestWilsonHRParameters:
    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_threshold(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["v", "r", "v_peak"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_state_and_threshold(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_r", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_r", "dt"])
    @pytest.mark.parametrize("value", [object(), "0.1", True])
    def test_rejects_non_numeric_scales(self, field: str, value: object):
        with pytest.raises(TypeError, match=field):
            WilsonHRNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(current)
        assert (n.v, n.r) == before

    @pytest.mark.parametrize("current", [object(), "0.3", True])
    def test_rejects_non_numeric_current_before_state_mutation(self, current: object):
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(TypeError, match="current"):
            n.step(current)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WilsonHRNeuron()
        n.r = np.inf
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="r must be finite"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_scale_before_mutation(self):
        n = WilsonHRNeuron()
        n.tau_r = 0.0
        before = (n.v, n.r)
        with pytest.raises(ValueError, match="tau_r"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_polynomial_overflow_before_state_mutation(self):
        n = WilsonHRNeuron(v=1.0e308)
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="polynomial|candidate|derivative"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_direct_derivative_rejects_non_finite_state(self):
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="state and current"):
            n._derivatives(np.nan, n.r, 0.3)

    def test_direct_derivative_rejects_non_finite_output(self):
        n = WilsonHRNeuron()
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(0.0, 1.0e308, 0.3)

    def test_direct_candidate_validation_rejects_non_finite_candidate(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            WilsonHRNeuron._validate_candidate(np.nan, 0.0)

    def test_tau_r_affects_recovery(self):
        n_fast = WilsonHRNeuron(tau_r=1.0)
        n_slow = WilsonHRNeuron(tau_r=5.0)
        s_fast = len(_run(n_fast, current=0.3, steps=50_000))
        s_slow = len(_run(n_slow, current=0.3, steps=50_000))
        assert s_fast != s_slow

    def test_v_peak_controls_threshold(self):
        n_low = WilsonHRNeuron(v_peak=0.2)
        n_high = WilsonHRNeuron(v_peak=0.6)
        s_low = len(_run(n_low, current=0.3, steps=50_000))
        s_high = len(_run(n_high, current=0.3, steps=50_000))
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = WilsonHRNeuron(dt=dt)
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WilsonHRNeuron()
            trace = [(n.step(0.3), n.v, n.r) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestWilsonHRPerformance:
    def test_isolation_throughput(self):
        n = WilsonHRNeuron()
        steps = 50_000
        t0 = time.perf_counter()
        for _ in range(steps):
            n.step(0.3)
        elapsed = time.perf_counter() - t0
        assert steps / elapsed > 10_000

    def test_network_throughput(self):
        pop = Population(WilsonHRNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5_000


class TestWilsonHRPublicWorkflow:
    """Named workflow contract: Wilson-HR public surface inside the Python simulator."""

    def test_population(self):
        assert Population(WilsonHRNeuron, n=10, label="whr").n == 10

    def test_network_spikes(self):
        pop = Population(WilsonHRNeuron, n=10, label="whr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(WilsonHRNeuron, n=10, label="src")
        tgt = Population(WilsonHRNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.2, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = WilsonHRNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(50_000)])
        sc = spike_count(train)
        assert sc >= 4
        intervals = isi(train, dt=0.00005)
        assert len(intervals) >= 3
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
