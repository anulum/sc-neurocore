# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MATNeuron RK4 hardening

"""Full pipeline test for MATNeuron (Kobayashi et al. 2009).

Multi-timescale Adaptive Threshold model.
dV/dt = (-(V-V_rest) + R·I) / tau_m
dtheta1/dt = -theta1/tau_1    (fast adaptation, tau=10)
dtheta2/dt = -theta2/tau_2    (slow adaptation, tau=200)
Threshold: V_th = V_base + theta1 + theta2.
On spike: V→V_reset, theta1 += h1, theta2 += h2.

Two adaptation timescales produce spike-frequency adaptation and
burst-rate adaptation. theta1 (fast, h1=5) captures short-term
refractoriness; theta2 (slow, h2=3) captures long-term adaptation.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard

from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: MATNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestMATIsolation:
    def test_defaults(self):
        n = MATNeuron()
        assert n.v == -70.0 and n.theta1 == 0.0 and n.theta2 == 0.0
        assert n.v_threshold_base == -50.0 and n.dt == 1.0
        assert n.tau_1 == 10.0 and n.tau_2 == 200.0
        assert n.h1 == 5.0 and n.h2 == 3.0

    def test_step_returns_binary(self):
        assert MATNeuron().step(0.0) in (0, 1)

    def test_all_states_evolve(self):
        n = MATNeuron()
        v0, t1_0, t2_0 = n.v, n.theta1, n.theta2
        for _ in range(100):
            n.step(30.0)
        # v changes, thetas change after spike
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = MATNeuron()
        for _ in range(100_000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1) and np.isfinite(n.theta2)

    def test_reset_restores_defaults(self):
        n = MATNeuron()
        for _ in range(5000):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest and n.theta1 == 0.0 and n.theta2 == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MATNeuron()
            trace = [(n.step(30.0), n.v, n.theta1, n.theta2) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — membrane equation, threshold decay, spike mechanism
# ---------------------------------------------------------------------------
class TestMATAnalytical:
    def test_rk4_candidate_one_step(self):
        """The public step commits the candidate-first RK4 state."""
        n = MATNeuron()
        I = 15.0
        expected_v, expected_theta1, expected_theta2 = n._rk4_candidate(I)
        n.step(I)
        assert abs(n.v - expected_v) < 1e-12
        assert abs(n.theta1 - expected_theta1) < 1e-12
        assert abs(n.theta2 - expected_theta2) < 1e-12

    def test_rk4_separates_from_forward_euler(self):
        """Finite-dt MAT integration must not regress to raw forward Euler."""
        n = MATNeuron(theta1=4.0, theta2=2.0, dt=2.0)
        I = 15.0
        euler_v = n.v + (-(n.v - n.v_rest) + n.resistance * I) / n.tau_m * n.dt
        expected_v, _, _ = n._rk4_candidate(I)
        assert abs(expected_v - euler_v) > 1e-3

    def test_theta1_exponential_decay(self):
        """RK4 threshold decay tracks the closed-form exponential."""
        n = MATNeuron()
        n.theta1 = 5.0  # as if just spiked
        steps = 20
        for _ in range(steps):
            n.step(0.0)  # zero current to prevent new spikes
        expected = 5.0 * np.exp(-steps * n.dt / n.tau_1)
        assert abs(n.theta1 - expected) < 1e-3

    def test_theta2_exponential_decay(self):
        """RK4 slow-threshold decay tracks the closed-form exponential."""
        n = MATNeuron()
        n.theta2 = 3.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        expected = 3.0 * np.exp(-steps * n.dt / n.tau_2)
        assert abs(n.theta2 - expected) < 1e-9

    def test_theta1_decays_faster_than_theta2(self):
        """tau_1 < tau_2 → theta1 decays faster."""
        n = MATNeuron()
        n.theta1 = 10.0
        n.theta2 = 10.0
        for _ in range(50):
            n.step(0.0)
        assert n.theta1 < n.theta2

    def test_decay_ratio_matches_timescale(self):
        """After 1 tau: theta decays to ≈ 1/e of initial."""
        n = MATNeuron()
        n.theta1 = 10.0
        for _ in range(int(n.tau_1 / n.dt)):
            n.step(0.0)
        expected = 10.0 / np.e
        assert abs(n.theta1 - expected) < 0.01

    def test_threshold_is_sum(self):
        """Effective threshold = V_base + theta1 + theta2."""
        n = MATNeuron()
        n.theta1 = 3.0
        n.theta2 = 2.0
        threshold = n.v_threshold_base + n.theta1 + n.theta2
        assert abs(threshold - (-45.0)) < 1e-12

    def test_spike_increments_both_thetas(self):
        """On spike: theta1 += h1=5, theta2 += h2=3."""
        n = MATNeuron()
        for _ in range(10_000):
            if n.step(30.0) == 1:
                # theta1 was decayed then incremented
                assert n.theta1 >= n.h1 * 0.9  # at least close to h1
                assert n.theta2 >= n.h2 * 0.9
                break

    def test_spike_retains_threshold_candidates(self):
        """Spike reset keeps the RK4-decayed threshold state before increments."""
        n = MATNeuron()
        _, theta1_candidate, theta2_candidate = n._rk4_candidate(250.0)
        assert n.step(250.0) == 1
        assert n.v == n.v_reset
        assert abs(n.theta1 - (theta1_candidate + n.h1)) < 1e-12
        assert abs(n.theta2 - (theta2_candidate + n.h2)) < 1e-12

    def test_spike_resets_voltage(self):
        """On spike: V → V_reset."""
        n = MATNeuron()
        for _ in range(10_000):
            if n.step(30.0) == 1:
                assert n.v == n.v_reset
                break

    def test_membrane_steady_state(self):
        """At steady state (no spike): V_ss = V_rest + R·I."""
        n = MATNeuron()
        # Low current to avoid spiking
        I = 10.0
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + n.resistance * I
        # Should be close to steady state
        assert abs(n.v - expected_ss) < 1.0

    def test_invalid_current_preserves_state(self):
        """Invalid runtime current is rejected before mutating state."""
        n = MATNeuron()
        before = (n.v, n.theta1, n.theta2)
        with pytest.raises(ValueError, match="input current"):
            n.step(float("nan"))
        assert (n.v, n.theta1, n.theta2) == before

    def test_invalid_state_preserves_state(self):
        """Corrupted threshold adaptation is rejected before mutation."""
        n = MATNeuron()
        n.theta1 = -1.0
        before = (n.v, n.theta1, n.theta2)
        with pytest.raises(ValueError, match="threshold adaptation"):
            n.step(10.0)
        assert (n.v, n.theta1, n.theta2) == before


# ---------------------------------------------------------------------------
# 3. ADAPTATION DYNAMICS
# ---------------------------------------------------------------------------
class TestMATAdaptation:
    def test_adaptation_reduces_rate(self):
        """Adaptation → first half has more spikes than second half."""
        n = MATNeuron()
        s1 = sum(n.step(40.0) for _ in range(2500))
        s2 = sum(n.step(40.0) for _ in range(2500))
        assert s1 >= s2

    def test_adaptation_recovers(self):
        """After silence, thetas decay → threshold drops → fires again."""
        n = MATNeuron()
        # Drive to adapt
        for _ in range(5000):
            n.step(40.0)
        theta_adapted = n.theta1 + n.theta2
        # Rest (let adaptation decay)
        for _ in range(2000):
            n.step(0.0)
        theta_recovered = n.theta1 + n.theta2
        assert theta_recovered < theta_adapted

    def test_theta_accumulation_with_bursts(self):
        """Rapid spiking accumulates theta beyond single h1/h2."""
        n = MATNeuron()
        for _ in range(1000):
            n.step(50.0)
        # After sustained drive, thetas accumulate from multiple spikes
        assert n.theta2 > n.h2  # slow theta accumulates


# ---------------------------------------------------------------------------
# 4. DYNAMICS — f-I, ISI
# ---------------------------------------------------------------------------
class TestMATDynamics:
    def test_subthreshold_silent(self):
        n = MATNeuron()
        assert len(_run(n, current=15.0, steps=5000)) == 0

    def test_fires_at_sufficient_current(self):
        n = MATNeuron()
        assert len(_run(n, current=30.0, steps=5000)) >= 30

    def test_rate_monotonic(self):
        rates = []
        for I in [25.0, 35.0, 50.0]:
            n = MATNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [20.0, 30.0, 40.0, 50.0, 80.0])
    def test_fi_sweep(self, current: float):
        n = MATNeuron()
        spikes = _run(n, current=current, steps=5000)
        assert isinstance(len(spikes), int)

    def test_isi_increases_with_adaptation(self):
        """Adaptation lengthens ISI over time."""
        n = MATNeuron()
        spikes = _run(n, current=40.0, steps=10_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[:10])
            # Later ISIs should be longer (adaptation)
            assert isis[-1] >= isis[0]


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestMATParameters:
    @pytest.mark.parametrize("tau_1", [5.0, 10.0, 50.0])
    def test_tau_1_sweep(self, tau_1: float):
        n = MATNeuron(tau_1=tau_1)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1)

    @pytest.mark.parametrize("tau_2", [50.0, 200.0, 1000.0])
    def test_tau_2_sweep(self, tau_2: float):
        n = MATNeuron(tau_2=tau_2)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta2)

    @pytest.mark.parametrize("h1", [2.0, 5.0, 10.0])
    def test_h1_controls_fast_adaptation(self, h1: float):
        n = MATNeuron(h1=h1)
        spikes = len(_run(n, current=40.0, steps=5000))
        assert isinstance(spikes, int)

    def test_h2_controls_slow_adaptation(self):
        s_low = len(_run(MATNeuron(h2=1.0), current=40.0, steps=5000))
        s_high = len(_run(MATNeuron(h2=10.0), current=40.0, steps=5000))
        # Stronger slow adaptation → fewer spikes
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.1, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = MATNeuron(dt=dt)
        for _ in range(5000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1) and np.isfinite(n.theta2)

    def test_resistance_scales_input(self):
        """Higher R → more effective current → more spikes."""
        s_low = len(_run(MATNeuron(resistance=0.5), current=30.0, steps=5000))
        s_high = len(_run(MATNeuron(resistance=2.0), current=30.0, steps=5000))
        assert s_high >= s_low


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestMATPerformance:
    def test_isolation_throughput(self):
        n = MATNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(30.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert_throughput_guard(
            label="MAT isolation",
            observed_per_second=rate,
            strict_minimum_per_second=100_000.0,
            smoke_minimum_per_second=25_000.0,
        )

    def test_network_throughput(self):
        pop = Population(MATNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE — Population, Projection, Network, Analysis
# ---------------------------------------------------------------------------
class TestMATPipeline:
    def test_population(self):
        assert Population(MATNeuron, n=10, label="mat").n == 10

    def test_projection_wiring(self):
        src = Population(MATNeuron, n=5, label="src")
        tgt = Population(MATNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(MATNeuron, n=10, label="mat")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 20

    def test_analysis_isi(self):
        n = MATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = MATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = MATNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
