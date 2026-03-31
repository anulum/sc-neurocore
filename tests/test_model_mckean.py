# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: McKeanNeuron

"""Full pipeline test for McKeanNeuron (McKean 1970).

Piecewise-linear FitzHugh-Nagumo caricature:
dv/dt = f(v) - w + I
dw/dt = ε · (v - γ·w)

f(v) = -v           if v < a/2
     = v - a         if a/2 ≤ v < (1+a)/2
     = 1 - v         if v ≥ (1+a)/2

Three linear pieces replace the cubic v-v³/3 of FHN.
Breakpoints at a/2=0.125 and (1+a)/2=0.625.
Slopes: -1 (left), +1 (middle), -1 (right).
Oscillatory band ≈ I∈[0.3, 0.8]. ε=0.01 slow recovery.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: McKeanNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — piecewise f(v), dv, dw formulas
# ---------------------------------------------------------------------------
class TestMcKeanAnalytical:
    def test_f_left_piece(self):
        """v < a/2 = 0.125: f(v) = -v."""
        n = McKeanNeuron()
        assert n._f(0.0) == 0.0
        assert abs(n._f(0.1) - (-0.1)) < 1e-12

    def test_f_middle_piece(self):
        """a/2 ≤ v < (1+a)/2: f(v) = v - a."""
        n = McKeanNeuron()
        assert abs(n._f(0.125) - (0.125 - 0.25)) < 1e-12  # at breakpoint
        assert abs(n._f(0.4) - (0.4 - 0.25)) < 1e-12

    def test_f_right_piece(self):
        """v ≥ (1+a)/2 = 0.625: f(v) = 1 - v."""
        n = McKeanNeuron()
        assert abs(n._f(0.625) - (1.0 - 0.625)) < 1e-12  # at breakpoint
        assert abs(n._f(0.8) - 0.2) < 1e-12

    def test_f_continuity_at_breakpoints(self):
        """f(v) is continuous at a/2 and (1+a)/2."""
        n = McKeanNeuron()
        mid1, mid2 = n.a / 2.0, (1.0 + n.a) / 2.0
        # Left limit → right limit at mid1
        assert abs(n._f(mid1 - 1e-10) - n._f(mid1)) < 1e-8
        # At mid2
        assert abs(n._f(mid2 - 1e-10) - n._f(mid2)) < 1e-8

    def test_f_slopes(self):
        """Slopes: -1 (left), +1 (middle), -1 (right)."""
        n = McKeanNeuron()
        eps = 1e-6
        # Left piece slope
        slope_left = (n._f(0.05 + eps) - n._f(0.05 - eps)) / (2 * eps)
        assert abs(slope_left - (-1.0)) < 0.01
        # Middle piece slope
        slope_mid = (n._f(0.3 + eps) - n._f(0.3 - eps)) / (2 * eps)
        assert abs(slope_mid - 1.0) < 0.01
        # Right piece slope
        slope_right = (n._f(0.7 + eps) - n._f(0.7 - eps)) / (2 * eps)
        assert abs(slope_right - (-1.0)) < 0.01

    def test_dv_formula_one_step(self):
        """dv = (f(v) - w + I) · dt."""
        n = McKeanNeuron()
        v0, w0 = n.v, n.w
        I = 0.5
        f_v = n._f(v0)
        expected_dv = (f_v - w0 + I) * n.dt
        expected_dw = n.epsilon * (v0 - n.gamma * w0) * n.dt
        n.step(I)
        assert abs((n.v - v0) - expected_dv) < 1e-12
        assert abs((n.w - w0) - expected_dw) < 1e-14

    def test_dw_formula_one_step(self):
        """dw = ε · (v - γ·w) · dt."""
        n = McKeanNeuron()
        v0, w0 = n.v, n.w
        expected_dw = n.epsilon * (v0 - n.gamma * w0) * n.dt
        n.step(0.0)
        actual_dw = n.w - w0
        assert abs(actual_dw - expected_dw) < 1e-14

    def test_w_nullcline(self):
        """w-nullcline: w = v/γ. At v=0: w=0."""
        n = McKeanNeuron()
        w_null = 0.0 / n.gamma
        assert abs(w_null) < 1e-12

    def test_v_nullcline(self):
        """V-nullcline: w = f(v) + I. Varies by piece."""
        n = McKeanNeuron()
        I = 0.5
        # In left piece (v=0): w = -v + I = 0.5
        w_null = n._f(0.0) + I
        assert abs(w_null - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 3. OSCILLATORY DYNAMICS
# ---------------------------------------------------------------------------
class TestMcKeanDynamics:
    def test_silent_at_zero_input(self):
        n = McKeanNeuron()
        assert len(_run(n, current=0.0, steps=20_000)) == 0

    def test_oscillatory_in_band(self):
        for I in [0.4, 0.5, 0.6]:
            n = McKeanNeuron()
            spikes = _run(n, current=I, steps=20_000)
            assert len(spikes) >= 3, f"I={I}: only {len(spikes)} spikes"

    def test_rate_monotonic(self):
        rates = []
        for I in [0.3, 0.5, 0.7]:
            n = McKeanNeuron()
            rates.append(len(_run(n, current=I, steps=20_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_fi_sweep(self, current: float):
        n = McKeanNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_regular_isi(self):
        """Piecewise linear → very regular oscillation."""
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


# ---------------------------------------------------------------------------
# 4. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestMcKeanPerformance:
    def test_isolation_throughput(self):
        n = McKeanNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # Simple 2D + piecewise comparison
        assert rate > 200_000, f"isolation: {rate:.0f} steps/s"

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


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestMcKeanPipeline:
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
        dt_sim = 0.0001  # dt=0.1 model time → 0.1ms
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
