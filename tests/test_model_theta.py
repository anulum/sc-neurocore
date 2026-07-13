# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ThetaNeuron

"""Full pipeline test for ThetaNeuron (Ermentrout & Kopell 1986).

Canonical Type-I neuron on the unit circle: dθ/dt = (1-cosθ) + (1+cosθ)·I.
Mathematically equivalent to QIF via change of variables.
Analytical: ISI = π/√I (continuous time), f = √I/π Hz."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sc_neurocore.accel.theta as backends
from sc_neurocore.neurons.models.theta import ThetaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: ThetaNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _wrap_phase(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def _exact_theta_candidate(theta: float, current: float, dt: float) -> tuple[float, bool]:
    y = math.tan(theta / 2.0)
    if current > 0.0:
        root_i = math.sqrt(current)
        phase = math.atan(y / root_i)
        next_phase = phase + root_i * dt
        return _wrap_phase(
            2.0 * math.atan(root_i * math.tan(next_phase))
        ), next_phase >= math.pi / 2.0
    if current == 0.0:
        denominator = 1.0 - y * dt
        if abs(denominator) <= 1e-15:
            return -math.pi, True
        return _wrap_phase(2.0 * math.atan(y / denominator)), denominator <= 0.0

    root_i = math.sqrt(-current)
    if math.isclose(y, -root_i, rel_tol=0.0, abs_tol=1e-15):
        return theta, False
    ratio = (y - root_i) / (y + root_i)
    evolved = ratio * math.exp(2.0 * root_i * dt)
    denominator = 1.0 - evolved
    spiked = ratio < 1.0 <= evolved or abs(denominator) <= 1e-15
    if spiked and abs(denominator) <= 1e-15:
        return -math.pi, True
    return _wrap_phase(2.0 * math.atan(root_i * (1.0 + evolved) / denominator)), spiked


class TestThetaIsolation:
    def test_construction_defaults(self) -> None:
        n = ThetaNeuron()
        assert n.theta == 0.0
        assert n.dt == 0.01

    def test_step_returns_binary(self) -> None:
        assert ThetaNeuron().step(0.0) in (0, 1)

    def test_theta_evolves(self) -> None:
        n = ThetaNeuron()
        n.step(1.0)
        assert n.theta != 0.0

    def test_theta_wrapped_to_minus_pi_pi(self) -> None:
        """theta is wrapped to [-π, π] after each step."""
        n = ThetaNeuron()
        for _ in range(10000):
            n.step(5.0)
        assert -np.pi <= n.theta <= np.pi

    def test_state_finite_long_run(self) -> None:
        n = ThetaNeuron()
        for _ in range(100000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    def test_reset(self) -> None:
        n = ThetaNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.theta == 0.0


class TestThetaValidation:
    @pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_initial_phase(self, theta: float) -> None:
        with pytest.raises(ValueError, match="theta"):
            ThetaNeuron(theta=theta)

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt: float) -> None:
        with pytest.raises(ValueError, match="dt"):
            ThetaNeuron(dt=dt)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_phase_mutation(self, current: float) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.theta == before

    def test_initial_phase_is_wrapped_to_compact_circle(self) -> None:
        n = ThetaNeuron(theta=4.0 * np.pi + 0.5)
        assert -np.pi <= n.theta <= np.pi
        assert abs(n.theta - 0.5) < 1e-12

    def test_rejects_non_finite_exact_candidate_before_state_mutation(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=1.0e308)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0e308)
        assert n.theta == before

    @pytest.mark.parametrize("field", ["theta", "dt"])
    def test_rejects_corrupted_runtime_state_before_phase_mutation(self, field: str) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "theta":
            assert n.theta == before

    def test_rejects_runtime_dt_that_is_no_longer_positive(self) -> None:
        n = ThetaNeuron(theta=0.25)
        before = n.theta
        n.dt = 0.0
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        assert n.theta == before


class TestThetaBifurcation:
    """Saddle-node bifurcation at I=0 — same as QIF."""

    def test_negative_current_silent(self) -> None:
        """I<0 → stable fixed point. No spikes."""
        for I in [-1.0, -0.5]:
            n = ThetaNeuron()
            spikes = _run(n, current=I, steps=50000)
            assert len(spikes) == 0, f"I={I}: {len(spikes)} spikes"

    def test_zero_current_silent(self) -> None:
        """I=0 → theta stays at 0 (fixed point)."""
        n = ThetaNeuron()
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.theta) < 1e-10

    def test_positive_current_fires(self) -> None:
        """I>0 → periodic spiking."""
        n = ThetaNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        assert len(spikes) >= 50

    def test_continuous_onset(self) -> None:
        """Rate rises continuously from zero at I=0+ (Type-I)."""
        n01 = ThetaNeuron()
        n10 = ThetaNeuron()
        s01 = len(_run(n01, current=0.1, steps=100000))
        s10 = len(_run(n10, current=1.0, steps=100000))
        assert 0 < s01 < s10

    def test_fixed_point_at_negative_I(self) -> None:
        """At I=-0.5, theta should converge to stable FP: θ* = -arccos((1+I)/(1-I))."""
        # For I=-0.5: (1+I)/(1-I) = 0.5/1.5 = 1/3, θ* = -arccos(1/3) ≈ -1.231
        n = ThetaNeuron()
        for _ in range(100000):
            n.step(-0.5)
        theta_analytical = -np.arccos(1.0 / 3.0)
        assert abs(n.theta - theta_analytical) < 0.01, (
            f"theta={n.theta:.4f}, expected={theta_analytical:.4f}"
        )


class TestThetaAnalyticalISI:
    """ISI = π/√I (continuous time). ISI_steps = π/(√I · dt)."""

    @pytest.mark.parametrize("I", [0.5, 1.0, 2.0, 5.0])
    def test_isi_matches_analytical(self, I: float) -> None:
        """Measured ISI × dt should equal π/√I within 2%."""
        n = ThetaNeuron()
        spikes = _run(n, current=I, steps=100000)
        assert len(spikes) >= 10
        isis = np.diff(spikes[2:])
        measured_isi_time = np.mean(isis) * n.dt
        analytical_isi = np.pi / np.sqrt(I)
        rel_error = abs(measured_isi_time - analytical_isi) / analytical_isi
        assert rel_error < 0.02, (
            f"I={I}: ISI_time={measured_isi_time:.4f}, analytical={analytical_isi:.4f}, "
            f"error={rel_error:.4f}"
        )

    def test_near_constant_isi(self) -> None:
        """ISI is near-constant, with only discrete step quantisation jitter."""
        n = ThetaNeuron()
        spikes = _run(n, current=1.0, steps=50000)
        isis = np.diff(spikes[2:])
        unique_isis = np.unique(isis)
        assert len(unique_isis) <= 2, f"Too many ISI values: {unique_isis}"
        assert max(unique_isis) - min(unique_isis) <= 1, f"ISI jitter > 1: {unique_isis}"

    def test_sqrt_scaling(self) -> None:
        """f(4I)/f(I) ≈ 2 (since f ∝ √I)."""
        n1 = ThetaNeuron()
        n4 = ThetaNeuron()
        s1 = len(_run(n1, current=1.0, steps=100000))
        s4 = len(_run(n4, current=4.0, steps=100000))
        ratio = s4 / s1
        assert 1.8 < ratio < 2.2, f"f(4I)/f(I) = {ratio:.2f}, expected ~2.0"


class TestThetaPhaseSpace:
    """Phase dynamics on the unit circle."""

    def test_theta_traverses_full_circle(self) -> None:
        """At I>0, theta should cycle through [-π, π]."""
        n = ThetaNeuron()
        thetas = set()
        for _ in range(10000):
            n.step(1.0)
            thetas.add(round(n.theta, 1))
        # Should visit many distinct theta values
        assert len(thetas) > 20

    def test_spike_at_pi(self) -> None:
        """Spike is detected when the exact flow crosses π from below."""
        n = ThetaNeuron()
        for _ in range(50000):
            if n.step(1.0) == 1:
                # After spike, theta is wrapped. Just verify spike occurred
                return
        pytest.fail("No spike in 50k steps at I=1.0")

    def test_dynamics_equation(self) -> None:
        """Verify the tangent-half-angle exact constant-current flow."""
        n = ThetaNeuron(theta=1.0)
        expected, spiked = _exact_theta_candidate(n.theta, 2.0, n.dt)
        result = n.step(2.0)
        assert result == int(spiked)
        assert abs(n.theta - expected) < 1e-12

    def test_exact_positive_flow_separates_from_forward_euler(self) -> None:
        n = ThetaNeuron(theta=1.0, dt=0.2)
        current = 2.0
        euler = _wrap_phase(
            n.theta + ((1.0 - math.cos(n.theta)) + (1.0 + math.cos(n.theta)) * current) * n.dt
        )
        expected, spiked = _exact_theta_candidate(n.theta, current, n.dt)
        result = n.step(current)
        assert result == int(spiked)
        assert abs(n.theta - expected) < 1e-12
        assert abs(n.theta - euler) > 1e-4

    def test_exact_flow_reports_within_step_crossing(self) -> None:
        n = ThetaNeuron(theta=2.5, dt=1.0)
        expected, spiked = _exact_theta_candidate(n.theta, 1.0, n.dt)
        assert spiked
        assert n.step(1.0) == 1
        assert abs(n.theta - expected) < 1e-12

    def test_negative_current_stable_fixed_point_is_preserved(self) -> None:
        n = ThetaNeuron(theta=-math.pi / 2.0, dt=100.0)
        assert n.step(-1.0) == 0
        assert abs(n.theta + math.pi / 2.0) < 1e-12


class TestThetaParameters:
    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float) -> None:
        n = ThetaNeuron(dt=dt)
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    def test_dt_affects_isi_steps_not_time(self) -> None:
        """Finer dt → more steps per ISI, but ISI_time stays the same."""
        n1 = ThetaNeuron(dt=0.01)
        n2 = ThetaNeuron(dt=0.005)
        s1 = _run(n1, current=1.0, steps=100000)
        s2 = _run(n2, current=1.0, steps=200000)
        if len(s1) > 5 and len(s2) > 5:
            isi_time_1 = np.mean(np.diff(s1[2:])) * 0.01
            isi_time_2 = np.mean(np.diff(s2[2:])) * 0.005
            assert abs(isi_time_1 - isi_time_2) < 0.1


class TestThetaEdgeCases:
    def test_theta_wrapping_correct(self) -> None:
        """After large positive dtheta, theta stays in [-π, π]."""
        n = ThetaNeuron(theta=3.0, dt=0.5)
        n.step(10.0)  # large jump
        assert -np.pi <= n.theta <= np.pi

    def test_candidate_phase_is_validated_before_assignment(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=1.0e308)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0e308)
        assert n.theta == before

    def test_positive_flow_singularity_wraps_and_reports_crossing(self) -> None:
        n = ThetaNeuron(theta=0.0, dt=math.pi / 2.0)
        assert n.step(1.0) == 1
        assert n.theta == -math.pi

    def test_zero_current_singularity_wraps_and_reports_crossing(self) -> None:
        n = ThetaNeuron(theta=math.pi / 2.0, dt=1.0)
        assert n.step(0.0) == 1
        assert n.theta == -math.pi

    def test_negative_flow_exponential_overflow_is_rejected_without_mutation(self) -> None:
        n = ThetaNeuron(theta=0.25, dt=400.0)
        before = n.theta
        with pytest.raises(ValueError, match="exact-flow candidate"):
            n.step(-1.0)
        assert n.theta == before

    def test_negative_flow_singularity_wraps_and_reports_crossing(self) -> None:
        theta = 2.0
        y = math.tan(theta / 2.0)
        ratio = (y - 1.0) / (y + 1.0)
        n = ThetaNeuron(theta=theta, dt=-math.log(ratio) / 2.0)
        assert n.step(-1.0) == 1
        assert n.theta == -math.pi

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = ThetaNeuron()
            trace = [(n.step(2.0), n.theta) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestThetaPipeline:
    def test_population(self) -> None:
        assert Population(ThetaNeuron, n=10, label="theta").n == 10

    def test_network_with_drive(self) -> None:
        pop = Population(ThetaNeuron, n=10, label="theta")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_propagates(self) -> None:
        src = Population(ThetaNeuron, n=10, label="src")
        tgt = Population(ThetaNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=3.0, probability=0.5, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_tgt.count > 0

    def test_analysis_pipeline(self) -> None:
        n = ThetaNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 1.0


class TestThetaSimulate:
    """Engineering-verification surface for ``ThetaNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = ThetaNeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1
        assert n.theta == float(trace[-1])

    def test_simulate_rust_matches_python(self) -> None:
        assert backends._HAS_RUST
        py = ThetaNeuron()
        rs = ThetaNeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        assert np.array_equal(tr_py, tr_rs)

    def test_simulate_rust_rejects_non_default(self) -> None:
        assert backends._HAS_RUST
        n = ThetaNeuron(dt=0.02)
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
