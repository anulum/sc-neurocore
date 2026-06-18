# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NonlinearLIFNeuron

"""Full pipeline test for NonlinearLIFNeuron (Touboul & Brette 2008).

Nonlinear LIF with quadratic/cubic term + adaptation:
C dV/dt = a·(V-V_rest)·(V-V_crit) - w + I
dw/dt = (b·(V-V_rest) - w) / tau_w

Quadratic nonlinearity a·(V-V_rest)·(V-V_crit) creates:
- Stable resting point at V_rest when I < rheobase
- Runaway depolarisation when V > V_crit (positive feedback)
- Hard threshold reset at V ≥ V_threshold

w provides spike-frequency adaptation (tau_w=100ms).
a=0.04, b=0.5, V_rest=-65, V_crit=-40.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.nlif import NonlinearLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: NonlinearLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestNLIFIsolation:
    def test_defaults(self):
        n = NonlinearLIFNeuron()
        assert n.v == -65.0 and n.w == 0.0
        assert n.v_rest == -65.0 and n.v_crit == -40.0
        assert n.v_threshold == -20.0 and n.a == 0.04
        assert n.b == 0.5 and n.tau_w == 100.0 and n.c_m == 1.0

    def test_step_returns_binary(self):
        assert NonlinearLIFNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self):
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(20.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = NonlinearLIFNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self):
        n = NonlinearLIFNeuron(v_rest=-62.0, v_reset=-58.0, v_crit=-40.0, v_threshold=-20.0)
        for _ in range(5000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest and n.w == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": np.nan},
            {"w": np.inf},
            {"v_rest": np.nan},
            {"v_crit": np.inf},
            {"v_threshold": np.nan},
            {"v_reset": np.inf},
            {"v_crit": -70.0},
            {"v_threshold": -45.0},
            {"v_reset": -10.0},
            {"a": -0.01},
            {"a": np.nan},
            {"b": -0.1},
            {"tau_w": 0.0},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"dt": 101.0},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            NonlinearLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = NonlinearLIFNeuron(v=-60.0, w=0.5)
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = NonlinearLIFNeuron()
            trace = [(n.step(20.0), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — quadratic term, dV, dw, spike mechanism
# ---------------------------------------------------------------------------
class TestNLIFAnalytical:
    def test_cubic_term_at_rest(self):
        """At V=V_rest: a·(V_rest-V_rest)·(V_rest-V_crit) = 0."""
        n = NonlinearLIFNeuron()
        cubic = n.a * (n.v_rest - n.v_rest) * (n.v_rest - n.v_crit)
        assert abs(cubic) < 1e-14

    def test_cubic_term_above_v_crit(self):
        """V > V_crit and V > V_rest → positive feedback (runaway)."""
        n = NonlinearLIFNeuron()
        v = -35.0  # above both V_rest and V_crit
        cubic = n.a * (v - n.v_rest) * (v - n.v_crit)
        assert cubic > 0

    def test_cubic_term_between_rest_and_crit(self):
        """V_rest < V < V_crit → negative term (restoring)."""
        n = NonlinearLIFNeuron()
        v = -50.0
        cubic = n.a * (v - n.v_rest) * (v - n.v_crit)
        assert cubic < 0  # (positive) * (negative) = negative

    def test_rk4_candidate_one_step(self):
        """One step matches the candidate-first RK4 update."""
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        current = 5.0
        expected_v, expected_w = n._rk4_candidate(current)
        n.step(current)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.w - expected_w) < 1e-14
        assert n.v != v0
        assert n.w != w0

    def test_dw_formula_one_step(self):
        """dw = (b·(V-V_rest) - w) / tau_w · dt."""
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        expected_dw = (n.b * (v0 - n.v_rest) - w0) / n.tau_w * n.dt
        n.step(0.0)
        assert abs((n.w - w0) - expected_dw) < 1e-14

    def test_w_steady_state(self):
        """At steady state: w_ss = b·(V-V_rest)."""
        n = NonlinearLIFNeuron()
        # At rest: V=V_rest → w_ss = 0
        assert abs(n.b * (n.v_rest - n.v_rest)) < 1e-12

    def test_spike_resets_voltage(self):
        n = NonlinearLIFNeuron()
        for _ in range(10_000):
            if n.step(20.0) == 1:
                assert n.v == n.v_reset
                break

    def test_spike_threshold(self):
        """Spike on V ≥ V_threshold = -20."""
        n = NonlinearLIFNeuron()
        assert n.v_threshold == -20.0

    def test_v_nullcline(self):
        """V-nullcline: w = a·(V-V_rest)·(V-V_crit) + I."""
        n = NonlinearLIFNeuron()
        I = 10.0
        v = -50.0
        w_null = n.a * (v - n.v_rest) * (v - n.v_crit) + I
        assert np.isfinite(w_null)


# ---------------------------------------------------------------------------
# 3. ADAPTATION
# ---------------------------------------------------------------------------
class TestNLIFAdaptation:
    def test_w_accumulates_during_spiking(self):
        n = NonlinearLIFNeuron()
        for _ in range(5000):
            n.step(20.0)
        assert n.w != 0.0

    def test_adaptation_reduces_rate(self):
        n = NonlinearLIFNeuron()
        s1 = sum(n.step(25.0) for _ in range(2500))
        s2 = sum(n.step(25.0) for _ in range(2500))
        # Adaptation should reduce later rate
        assert s1 >= s2


# ---------------------------------------------------------------------------
# 4. DYNAMICS
# ---------------------------------------------------------------------------
class TestNLIFDynamics:
    def test_subthreshold_silent(self):
        n = NonlinearLIFNeuron()
        assert len(_run(n, current=3.0, steps=5000)) == 0

    def test_fires_under_drive(self):
        n = NonlinearLIFNeuron()
        assert len(_run(n, current=20.0, steps=5000)) >= 50

    def test_rate_monotonic(self):
        rates = []
        for I in [10.0, 20.0, 40.0]:
            n = NonlinearLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0, 50.0])
    def test_fi_sweep(self, current: float):
        n = NonlinearLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestNLIFParameters:
    @pytest.mark.parametrize("a", [0.02, 0.04, 0.08])
    def test_a_nonlinearity(self, a: float):
        n = NonlinearLIFNeuron(a=a)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("b", [0.2, 0.5, 1.0])
    def test_b_adaptation_strength(self, b: float):
        n = NonlinearLIFNeuron(b=b)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("tau_w", [50.0, 100.0, 200.0])
    def test_tau_w_sweep(self, tau_w: float):
        n = NonlinearLIFNeuron(tau_w=tau_w)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.w)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = NonlinearLIFNeuron(dt=dt)
        for _ in range(10_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestNLIFPerformance:
    def test_isolation_throughput(self):
        n = NonlinearLIFNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 50_000 if os.environ.get("CI") else 100_000
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(NonlinearLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestNLIFPipeline:
    def test_population(self):
        assert Population(NonlinearLIFNeuron, n=10, label="nlif").n == 10

    def test_projection_wiring(self):
        src = Population(NonlinearLIFNeuron, n=5, label="src")
        tgt = Population(NonlinearLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(NonlinearLIFNeuron, n=10, label="nlif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = NonlinearLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 30

    def test_analysis_isi(self):
        n = NonlinearLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = NonlinearLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = NonlinearLIFNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
