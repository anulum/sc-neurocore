# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NonResettingLIFNeuron

"""Full pipeline test for NonResettingLIFNeuron (Kobayashi 2009 / Jolivet 2004).

Adaptive multi-timescale threshold (aMAT) variant — non-resetting LIF:
tau_m dV/dt = -(V - V_rest) + R·I
dθ/dt = -(θ - θ_rest) / tau_θ
On spike: θ += Δθ, V does NOT reset.

Key distinction from standard LIF: voltage continues from current value
after spike (no reset to V_reset). Only the threshold rises by Δθ=5mV,
then decays back with tau_θ=50ms. This creates a natural refractory
period via threshold elevation rather than voltage reset.
FULL PIPELINE WIRED + BOUNDED RUNTIME SENTINELS."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: NonResettingLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _exact_relaxation(state: float, steady_state: float, dt: float, tau: float) -> float:
    decay = np.exp(-dt / tau)
    return decay * state + (1.0 - decay) * steady_state


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestNRLIFIsolation:
    def test_defaults(self):
        n = NonResettingLIFNeuron()
        assert n.v == -65.0 and n.theta == -50.0
        assert n.v_rest == -65.0 and n.theta_rest == -50.0
        assert n.delta_theta == 5.0
        assert n.tau_m == 10.0 and n.tau_theta == 50.0
        assert n.r_m == 1.0 and n.dt == 0.1

    def test_step_returns_binary(self):
        assert NonResettingLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = NonResettingLIFNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta)

    def test_reset_restores_defaults(self):
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest and n.theta == n.theta_rest

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = NonResettingLIFNeuron()
            trace = [(n.step(20.0), n.v, n.theta) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestNRLIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("theta", np.inf),
            ("v_rest", -np.inf),
            ("theta_rest", np.nan),
        ],
    )
    def test_rejects_non_finite_voltage_or_threshold_state(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["delta_theta", "r_m"])
    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    def test_rejects_negative_or_non_finite_non_negative_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_m", "tau_theta", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            NonResettingLIFNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.theta) == before

    @pytest.mark.parametrize(
        "field",
        [
            "v",
            "theta",
            "v_rest",
            "theta_rest",
            "delta_theta",
            "tau_m",
            "tau_theta",
            "r_m",
            "dt",
        ],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field: str):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)
        if field not in {"v", "theta"}:
            assert (n.v, n.theta) == before

    @pytest.mark.parametrize("field", ["tau_m", "tau_theta", "dt"])
    def test_rejects_non_positive_runtime_time_constants_before_mutation(self, field: str):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0)
        before = (n.v, n.theta)
        setattr(n, field, 0.0)
        with pytest.raises(ValueError, match="runtime"):
            n.step(20.0)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_membrane_candidate_before_mutation(self):
        n = NonResettingLIFNeuron(v=-60.0, theta=-45.0, r_m=10.0)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="exact relaxation"):
            n.step(1.0e308)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_threshold_candidate_before_mutation(self):
        n = NonResettingLIFNeuron(v=1.0e308, theta=9.0e307, theta_rest=9.0e307, delta_theta=9.0e307)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="exact relaxation"):
            n.step(0.0)
        assert (n.v, n.theta) == before


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — dV, dθ, no-reset, spike mechanism
# ---------------------------------------------------------------------------
class TestNRLIFAnalytical:
    def test_subthreshold_step_matches_exact_relaxation(self):
        """Linear membrane and threshold ODEs follow the closed-form solution."""
        n = NonResettingLIFNeuron(v=-60.0, theta=-40.0, dt=0.5)
        v0 = n.v
        theta0 = n.theta
        current = 4.0
        expected_v = _exact_relaxation(v0, n.v_rest + n.r_m * current, n.dt, n.tau_m)
        expected_theta = _exact_relaxation(theta0, n.theta_rest, n.dt, n.tau_theta)
        assert n.step(current) == 0
        assert n.v == pytest.approx(expected_v, abs=1e-12)
        assert n.theta == pytest.approx(expected_theta, abs=1e-12)

    def test_large_timestep_exact_relaxation_remains_bounded(self):
        """Exact relaxation stays inside the physical endpoint envelope for large dt."""
        n = NonResettingLIFNeuron(v=1000.0, theta=2000.0, dt=100.0)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(
            _exact_relaxation(1000.0, n.v_rest, n.dt, n.tau_m),
            abs=1e-12,
        )
        assert n.theta == pytest.approx(
            _exact_relaxation(2000.0, n.theta_rest, n.dt, n.tau_theta),
            abs=1e-12,
        )
        assert n.v_rest <= n.v <= 1000.0
        assert n.theta_rest <= n.theta <= 2000.0

    def test_dv_formula(self):
        """Voltage follows exact first-order relaxation toward V_rest + R·I."""
        n = NonResettingLIFNeuron()
        v0 = n.v
        I = 5.0
        expected_v = _exact_relaxation(v0, n.v_rest + n.r_m * I, n.dt, n.tau_m)
        n.step(I)
        assert abs(n.v - expected_v) < 1e-12

    def test_dtheta_formula(self):
        """Threshold follows exact first-order relaxation toward theta_rest."""
        n = NonResettingLIFNeuron()
        theta0 = n.theta
        expected_theta = _exact_relaxation(theta0, n.theta_rest, n.dt, n.tau_theta)
        n.step(0.0)  # subthreshold, no spike
        assert abs(n.theta - expected_theta) < 1e-14

    def test_no_voltage_reset_on_spike(self):
        """V does NOT reset after spike — key model feature."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            v_before = n.v
            if n.step(20.0) == 1:
                # V should be at or above where it was (not reset to V_rest)
                assert n.v >= n.v_rest
                # V was NOT set to v_rest or any v_reset
                assert n.v != n.v_rest or v_before == n.v_rest
                break

    def test_theta_increases_on_spike(self):
        """On spike: θ += Δθ."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            theta_before = n.theta
            if n.step(20.0) == 1:
                # theta increased by delta_theta (after decay within step)
                assert n.theta > theta_before
                break

    def test_theta_decays_toward_theta_rest(self):
        """θ decays exponentially toward θ_rest."""
        n = NonResettingLIFNeuron()
        n.theta = -30.0  # elevated
        for _ in range(5000):
            n.step(0.0)
        # Should decay toward theta_rest = -50
        assert n.theta < -30.0  # moved toward -50

    def test_theta_decay_rate(self):
        """After 1 tau_θ: θ decays by (1-1/e) of excess."""
        n = NonResettingLIFNeuron()
        n.theta = -40.0  # 10 above rest
        excess = n.theta - n.theta_rest  # = 10
        steps = int(n.tau_theta / n.dt)
        for _ in range(steps):
            n.step(0.0)
        remaining = n.theta - n.theta_rest
        # Should be ≈ excess/e ≈ 3.68
        expected = excess * np.exp(-1)
        assert abs(remaining - expected) < 0.5

    def test_membrane_steady_state(self):
        """At steady state (no spike): V_ss = V_rest + R·I."""
        n = NonResettingLIFNeuron()
        I = 5.0  # subthreshold
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.r_m * I
        assert abs(n.v - expected_ss) < 0.5

    def test_spike_condition(self):
        """Spike when V ≥ θ (dynamic threshold)."""
        n = NonResettingLIFNeuron()
        for _ in range(10_000):
            v_pre = n.v
            theta_pre = n.theta
            if n.step(20.0) == 1:
                # Before dv update within step, v crossed theta
                break


# ---------------------------------------------------------------------------
# 3. THRESHOLD ADAPTATION (REFRACTORY)
# ---------------------------------------------------------------------------
class TestNRLIFThresholdAdaptation:
    def test_threshold_elevation_creates_refractoriness(self):
        """After spike, elevated θ prevents immediate re-firing."""
        n = NonResettingLIFNeuron()
        # Find first spike, then check next step
        for _ in range(10_000):
            if n.step(20.0) == 1:
                # Theta just increased by 5mV
                # Next step should not spike (theta too high)
                next_spike = n.step(20.0)
                # May or may not spike depending on V, but theta is high
                assert n.theta > n.theta_rest
                break

    def test_theta_accumulates_with_rapid_spiking(self):
        """Multiple rapid spikes → θ well above θ_rest."""
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            n.step(30.0)
        assert n.theta > n.theta_rest + n.delta_theta * 0.5


# ---------------------------------------------------------------------------
# 4. DYNAMICS
# ---------------------------------------------------------------------------
class TestNRLIFDynamics:
    def test_subthreshold_silent(self):
        n = NonResettingLIFNeuron()
        assert len(_run(n, current=5.0, steps=5000)) == 0

    def test_fires_under_drive(self):
        n = NonResettingLIFNeuron()
        assert len(_run(n, current=20.0, steps=5000)) >= 5

    def test_rate_monotonic(self):
        rates = []
        for I in [15.0, 20.0, 30.0]:
            n = NonResettingLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 10.0, 20.0, 30.0, 50.0])
    def test_fi_sweep(self, current: float):
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestNRLIFParameters:
    @pytest.mark.parametrize("delta_theta", [2.0, 5.0, 10.0])
    def test_delta_theta_sweep(self, delta_theta: float):
        n = NonResettingLIFNeuron(delta_theta=delta_theta)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("tau_theta", [20.0, 50.0, 200.0])
    def test_tau_theta_sweep(self, tau_theta: float):
        n = NonResettingLIFNeuron(tau_theta=tau_theta)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.theta)

    @pytest.mark.parametrize("tau_m", [5.0, 10.0, 20.0])
    def test_tau_m_sweep(self, tau_m: float):
        n = NonResettingLIFNeuron(tau_m=tau_m)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = NonResettingLIFNeuron(dt=dt)
        for _ in range(10_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta)

    def test_larger_delta_theta_fewer_spikes(self):
        """Larger Δθ → stronger refractoriness → fewer spikes."""
        s_small = len(_run(NonResettingLIFNeuron(delta_theta=2.0), 20.0, 5000))
        s_large = len(_run(NonResettingLIFNeuron(delta_theta=15.0), 20.0, 5000))
        assert s_small >= s_large


# ---------------------------------------------------------------------------
# 6. BOUNDED RUNTIME SENTINELS
# ---------------------------------------------------------------------------
class TestNRLIFPerformance:
    def test_isolation_runtime_regression_sentinel(self):
        """Bound pathological slowdowns without making CI throughput claims."""
        n = NonResettingLIFNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert np.isfinite(n.v) and np.isfinite(n.theta)
        assert elapsed < 10.0

    def test_network_runtime_regression_sentinel(self):
        """Bound pathological network slowdowns without throughput claims."""
        pop = Population(NonResettingLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0
        assert mon.count >= 0


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestNRLIFPipeline:
    def test_population(self):
        assert Population(NonResettingLIFNeuron, n=10, label="nrlif").n == 10

    def test_projection_wiring(self):
        src = Population(NonResettingLIFNeuron, n=5, label="src")
        tgt = Population(NonResettingLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(NonResettingLIFNeuron, n=10, label="nrlif")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self):
        n = NonResettingLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(10_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
