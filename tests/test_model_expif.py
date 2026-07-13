# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ExpIFNeuron

"""Source-fidelity and pipeline tests for the maintained ExpIF neuron.

The voltage flow follows Fourcaud-Trocmé et al. (2003), Equations 6 and
10, after division by leak conductance. ``v_rh`` is the soft exponential
threshold; ``v_threshold`` is a separate finite spike cutoff.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, isi, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.expif import ExpIFNeuron


def _run(neuron: ExpIFNeuron, current: float, steps: int) -> list[int]:
    """Return the step indices at which the neuron emits a spike."""
    return [index for index in range(steps) if neuron.step(current) == 1]


def _rhs(neuron: ExpIFNeuron, v: float, current: float) -> float:
    """Independent source-equation derivative with the event-surface bound."""
    bounded_v = min(v, neuron.v_threshold)
    exponential = neuron.delta_t * math.exp((bounded_v - neuron.v_rh) / neuron.delta_t)
    return (-(bounded_v - neuron.v_rest) + exponential + current) / neuron.tau


def _rk4_candidate(neuron: ExpIFNeuron, current: float) -> float:
    """Independent candidate-first classical RK4 update."""
    v0 = neuron.v
    k1 = _rhs(neuron, v0, current)
    k2 = _rhs(neuron, v0 + 0.5 * neuron.dt * k1, current)
    k3 = _rhs(neuron, v0 + 0.5 * neuron.dt * k2, current)
    k4 = _rhs(neuron, v0 + neuron.dt * k3, current)
    return v0 + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler_candidate(neuron: ExpIFNeuron, current: float) -> float:
    """Return raw Euler solely to prove the maintained method is not Euler."""
    return neuron.v + neuron.dt * _rhs(neuron, neuron.v, current)


class TestExpIFIsolation:
    def test_construction_uses_source_fitted_defaults(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron.v == -65.0
        assert neuron.v_rest == -65.0
        assert neuron.v_reset == -68.0
        assert neuron.v_threshold == 30.0
        assert neuron.v_rh == -59.9
        assert neuron.delta_t == 3.48
        assert neuron.tau == 10.0
        assert neuron.dt == 0.02
        assert neuron.refractory_period == 0.0
        assert neuron.refractory_remaining == 0.0

    def test_step_returns_binary(self) -> None:
        assert ExpIFNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self) -> None:
        neuron = ExpIFNeuron()
        initial = neuron.v
        neuron.step(20.0)
        assert neuron.v != initial

    def test_state_remains_finite_and_below_cutoff(self) -> None:
        neuron = ExpIFNeuron()
        for _ in range(50_000):
            neuron.step(20.0)
        assert math.isfinite(neuron.v)
        assert neuron.v < neuron.v_threshold

    def test_reset_restores_rest_and_clears_refractory_state(self) -> None:
        neuron = ExpIFNeuron(refractory_period=0.06)
        neuron.v = 29.0
        assert neuron.step(0.0) == 1
        assert neuron.refractory_remaining == pytest.approx(0.06)
        neuron.reset()
        assert neuron.v == neuron.v_rest
        assert neuron.refractory_remaining == 0.0


class TestExpIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", math.nan),
            ("v_rest", math.inf),
            ("v_reset", -math.inf),
            ("v_threshold", math.nan),
            ("v_rh", math.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            ExpIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["delta_t", "tau", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite_scales(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            ExpIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["refractory_period", "refractory_remaining"])
    @pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
    def test_rejects_invalid_refractory_values(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            ExpIFNeuron(**{field: value})

    def test_rejects_inconsistent_threshold_relationships(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            ExpIFNeuron(v_threshold=-60.0, v_rh=-59.9)
        with pytest.raises(ValueError, match="below v_threshold"):
            ExpIFNeuron(v=30.0)
        with pytest.raises(ValueError, match="below v_threshold"):
            ExpIFNeuron(v_reset=31.0)

    def test_rejects_refractory_remainder_above_period(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ExpIFNeuron(refractory_period=0.02, refractory_remaining=0.04)

    @pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        neuron = ExpIFNeuron(v=-60.0)
        before = (neuron.v, neuron.refractory_remaining)
        with pytest.raises(ValueError, match="current"):
            neuron.step(current)
        assert (neuron.v, neuron.refractory_remaining) == before

    @pytest.mark.parametrize("runtime_v", [math.nan, 30.0, math.inf])
    def test_rejects_invalid_runtime_voltage_before_update(self, runtime_v: float) -> None:
        neuron = ExpIFNeuron(v=-60.0)
        neuron.v = runtime_v
        with pytest.raises(ValueError, match="runtime voltage state"):
            neuron.step(0.0)

    def test_rejects_invalid_runtime_refractory_state_before_update(self) -> None:
        neuron = ExpIFNeuron(refractory_period=0.02)
        neuron.refractory_remaining = 0.03
        with pytest.raises(ValueError, match="runtime refractory state"):
            neuron.step(0.0)
        assert neuron.refractory_remaining == 0.03

    def test_rejects_non_finite_rk4_candidate_before_state_mutation(self) -> None:
        neuron = ExpIFNeuron(v=-60.0, dt=1.0e308, tau=1.0)
        before = neuron.v
        with pytest.raises(ValueError, match="RK4"):
            neuron.step(1.0e308)
        assert neuron.v == before

    def test_rejects_overflowing_exponential_stage(self) -> None:
        neuron = ExpIFNeuron(v_threshold=1.0e300, v_rh=0.0, delta_t=1.0)
        with pytest.raises(ValueError, match="exponential term"):
            neuron._rhs(neuron.v_threshold, 0.0)

    def test_rejects_non_finite_derivative(self) -> None:
        neuron = ExpIFNeuron(tau=1.0e-308)
        with pytest.raises(ValueError, match="derivative"):
            neuron._rhs(neuron.v, 1.0e308)


class TestExpIFExponentialEscape:
    def test_exponential_term_at_soft_threshold_equals_delta_t(self) -> None:
        neuron = ExpIFNeuron()
        exponential = neuron.delta_t * math.exp((neuron.v_rh - neuron.v_rh) / neuron.delta_t)
        assert exponential == neuron.delta_t

    def test_hard_cutoff_is_distinct_from_soft_threshold(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron.v_threshold > neuron.v_rh + 20.0 * neuron.delta_t

    def test_rk4_stages_are_bounded_only_at_the_event_surface(self) -> None:
        neuron = ExpIFNeuron()
        assert neuron._rhs(1.0e9, 7.0) == neuron._rhs(neuron.v_threshold, 7.0)
        assert neuron._rhs(neuron.v_threshold - 1.0, 7.0) != neuron._rhs(neuron.v_threshold, 7.0)

    def test_candidate_crossing_cutoff_emits_and_resets(self) -> None:
        neuron = ExpIFNeuron(v=29.0)
        assert neuron.step(0.0) == 1
        assert neuron.v == neuron.v_reset

    def test_delta_t_controls_spike_initiation(self) -> None:
        sharp = len(_run(ExpIFNeuron(delta_t=0.5), current=20.0, steps=10_000))
        broad = len(_run(ExpIFNeuron(delta_t=5.0), current=20.0, steps=10_000))
        assert sharp != broad

    def test_negative_extreme_remains_finite(self) -> None:
        neuron = ExpIFNeuron(v=-1000.0)
        for _ in range(100):
            neuron.step(0.0)
        assert math.isfinite(neuron.v)


class TestExpIFAnalytical:
    def test_one_step_matches_independent_rk4(self) -> None:
        neuron = ExpIFNeuron(v=-62.0, dt=0.02)
        expected = _rk4_candidate(neuron, 5.0)
        assert neuron.step(5.0) == 0
        assert neuron.v == pytest.approx(expected, abs=1.0e-12)

    def test_rk4_separates_from_raw_euler_near_onset(self) -> None:
        neuron = ExpIFNeuron(v=-56.0, dt=0.2)
        rk4 = _rk4_candidate(neuron, 12.0)
        euler = _euler_candidate(neuron, 12.0)
        assert abs(rk4 - euler) > 1.0e-4
        assert neuron.step(12.0) == 0
        assert neuron.v == pytest.approx(rk4, abs=1.0e-12)

    def test_zero_current_relaxes_to_source_equilibrium(self) -> None:
        neuron = ExpIFNeuron()
        for _ in range(10_000):
            neuron.step(0.0)
        assert abs(neuron.v - neuron.v_rest) < 1.2
        assert neuron.v == pytest.approx(-63.896297890416314, abs=1.0e-10)

    def test_refractory_hold_is_discrete_and_deterministic(self) -> None:
        neuron = ExpIFNeuron(v=29.0, refractory_period=0.06)
        assert neuron.step(0.0) == 1
        for _ in range(3):
            assert neuron.step(100.0) == 0
            assert neuron.v == neuron.v_reset
        assert neuron.refractory_remaining == 0.0
        neuron.step(100.0)
        assert neuron.v != neuron.v_reset


class TestExpIFFI:
    @pytest.mark.parametrize(
        ("current", "expected"),
        [(0.0, 0), (5.0, 0), (10.0, 1), (20.0, 2), (50.0, 5), (100.0, 9)],
    )
    def test_enrolled_1000_step_event_goldens(self, current: float, expected: int) -> None:
        assert len(_run(ExpIFNeuron(), current=current, steps=1000)) == expected

    def test_subthreshold_current_is_silent(self) -> None:
        assert _run(ExpIFNeuron(), current=1.0, steps=10_000) == []

    def test_suprathreshold_current_fires(self) -> None:
        assert len(_run(ExpIFNeuron(), current=20.0, steps=10_000)) == 23

    def test_monotonic_fi_on_enrolled_operating_points(self) -> None:
        counts = [
            len(_run(ExpIFNeuron(), current=current, steps=1000))
            for current in (0.0, 5.0, 10.0, 20.0, 50.0, 100.0)
        ]
        assert counts == [0, 0, 1, 2, 5, 9]

    def test_constant_drive_has_regular_interspike_intervals(self) -> None:
        spikes = _run(ExpIFNeuron(), current=50.0, steps=10_000)
        intervals = np.diff(spikes[3:]).astype(float)
        assert intervals.size > 10
        assert float(np.std(intervals) / np.mean(intervals)) < 0.05


class TestExpIFParameters:
    def test_tau_affects_rate(self) -> None:
        fast = len(_run(ExpIFNeuron(tau=5.0), current=20.0, steps=10_000))
        slow = len(_run(ExpIFNeuron(tau=40.0), current=20.0, steps=10_000))
        assert fast > slow

    def test_lower_soft_threshold_fires_more_readily(self) -> None:
        lower = len(_run(ExpIFNeuron(v_rh=-62.0), current=15.0, steps=10_000))
        higher = len(_run(ExpIFNeuron(v_rh=-55.0), current=15.0, steps=10_000))
        assert lower > higher

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float) -> None:
        neuron = ExpIFNeuron(dt=dt)
        for _ in range(10_000):
            neuron.step(20.0)
        assert math.isfinite(neuron.v)
        assert neuron.v < neuron.v_threshold

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            neuron = ExpIFNeuron()
            traces.append([(neuron.step(20.0), neuron.v) for _ in range(1000)])
        assert traces[0] == traces[1]


class TestExpIFPerformance:
    def test_isolation_throughput(self) -> None:
        neuron = ExpIFNeuron()
        steps = 50_000
        started = time.perf_counter()
        for _ in range(steps):
            neuron.step(20.0)
        rate = steps / (time.perf_counter() - started)
        minimum = 10_000 if os.getenv("CI") else 12_000
        assert rate > minimum, f"local RK4 regression: {rate:.0f} steps/s, minimum={minimum}"

    def test_network_throughput(self) -> None:
        population = Population(ExpIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        monitor = SpikeMonitor(population)
        network = Network(population, drive, monitor)
        started = time.perf_counter()
        network.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - started
        assert 50 * 500 / elapsed > 5000


class TestExpIFPipeline:
    def test_population(self) -> None:
        assert Population(ExpIFNeuron, n=10, label="expif").n == 10

    def test_network_spikes(self) -> None:
        population = Population(ExpIFNeuron, n=10, label="expif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        monitor = SpikeMonitor(population)
        network = Network(population, drive, monitor)
        network.run(duration=1.0, dt=0.001, backend="python")
        assert monitor.count > 0

    def test_projection_wiring(self) -> None:
        source = Population(ExpIFNeuron, n=10, label="src")
        target = Population(ExpIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        projection = Projection(source, target, weight=20.0, probability=1.0, seed=42)
        monitor = SpikeMonitor(source)
        network = Network(source, target, drive, projection, monitor)
        network.run(duration=1.0, dt=0.001, backend="python")
        assert monitor.count > 0

    def test_analysis_pipeline(self) -> None:
        neuron = ExpIFNeuron()
        train = np.array([float(neuron.step(50.0)) for _ in range(10_000)])
        assert spike_count(train) == 52
        assert len(isi(train, dt=0.00002)) >= 5
        assert firing_rate(train, dt=0.00002) > 0.0
