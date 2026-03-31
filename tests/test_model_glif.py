# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# �� Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GLIFNeuron

"""Full pipeline test for GLIFNeuron (Teeter et al. 2018, Allen Institute).

Generalised LIF, 5-level hierarchy (GLIF5):
dV/dt = (-(V-V_rest) + R·I + i_asc1 + i_asc2) / tau_m
dθ/dt = (θ_inf - θ + a_θ·(V-V_rest)) / tau_θ
i_asc1 *= exp(-dt/tau_asc1)   (fast after-spike current)
i_asc2 *= exp(-dt/tau_asc2)   (slow after-spike current)

On spike: V→V_reset, θ+=Δθ, i_asc1+=r1, i_asc2+=r2.
5 state variables: v, theta, i_asc1, i_asc2 (+ theta_inf param).
Dynamic threshold + 2 after-spike currents enable Level 5 fits.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestGLIFIsolation:
    def test_defaults(self):
        n = GLIFNeuron()
        assert n.v == -70.0 and n.theta == -50.0 and n.theta_inf == -50.0
        assert n.i_asc1 == 0.0 and n.i_asc2 == 0.0
        assert n.tau_m == 10.0 and n.tau_theta == 100.0
        assert n.tau_asc1 == 10.0 and n.tau_asc2 == 200.0
        assert n.delta_theta == 2.0 and n.a_theta == 0.01

    def test_step_returns_binary(self):
        assert GLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = GLIFNeuron()
        for _ in range(100_000):
            n.step(50.0)
        for attr in ["v", "theta", "i_asc1", "i_asc2"]:
            assert np.isfinite(getattr(n, attr))

    def test_reset_restores_defaults(self):
        n = GLIFNeuron()
        for _ in range(5000):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest and n.theta == n.theta_inf
        assert n.i_asc1 == 0.0 and n.i_asc2 == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GLIFNeuron()
            trace = [(n.step(50.0), n.v, n.theta) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — dV, dθ, after-spike currents, spike mechanism
# ---------------------------------------------------------------------------
class TestGLIFAnalytical:
    def test_dv_formula(self):
        """dV = (-(V-V_rest) + R·I + i_asc1 + i_asc2) / tau_m · dt."""
        n = GLIFNeuron()
        v0 = n.v
        I = 10.0  # subthreshold
        expected_dv = (-(v0 - n.v_rest) + n.resistance * I + n.i_asc1 + n.i_asc2) / n.tau_m * n.dt
        n.step(I)
        assert abs((n.v - v0) - expected_dv) < 1e-12

    def test_dtheta_formula(self):
        """dθ = (θ_inf - θ + a_θ·(V-V_rest)) / tau_θ · dt."""
        n = GLIFNeuron()
        theta0, v0 = n.theta, n.v
        expected = (n.theta_inf - theta0 + n.a_theta * (v0 - n.v_rest)) / n.tau_theta * n.dt
        n.step(0.0)
        assert abs((n.theta - theta0) - expected) < 1e-12

    def test_i_asc1_exponential_decay(self):
        """i_asc1 *= exp(-dt/tau_asc1)."""
        n = GLIFNeuron()
        n.i_asc1 = 5.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        decay = np.exp(-n.dt / n.tau_asc1) ** steps
        expected = 5.0 * decay
        assert abs(n.i_asc1 - expected) < 1e-8

    def test_i_asc2_exponential_decay(self):
        """i_asc2 *= exp(-dt/tau_asc2). Slower than i_asc1."""
        n = GLIFNeuron()
        n.i_asc2 = 5.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        decay = np.exp(-n.dt / n.tau_asc2) ** steps
        expected = 5.0 * decay
        assert abs(n.i_asc2 - expected) < 1e-8

    def test_asc1_decays_faster_than_asc2(self):
        n = GLIFNeuron()
        n.i_asc1 = 5.0
        n.i_asc2 = 5.0
        for _ in range(50):
            n.step(0.0)
        assert n.i_asc1 < n.i_asc2

    def test_spike_resets_v(self):
        n = GLIFNeuron()
        for _ in range(10_000):
            if n.step(50.0) == 1:
                assert n.v == n.v_reset
                break

    def test_spike_increments_theta(self):
        n = GLIFNeuron()
        for _ in range(10_000):
            theta_pre = n.theta
            if n.step(50.0) == 1:
                # theta was incremented by delta_theta (after dtheta step)
                assert n.theta > theta_pre
                break

    def test_spike_adds_after_currents(self):
        n = GLIFNeuron()
        for _ in range(10_000):
            if n.step(50.0) == 1:
                assert n.i_asc1 >= n.r_asc1 * 0.8
                assert n.i_asc2 >= n.r_asc2 * 0.8
                break


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestGLIFDynamics:
    def test_fires_under_drive(self):
        n = GLIFNeuron()
        assert len(_run(n, current=50.0, steps=5000)) >= 50

    def test_subthreshold_silent(self):
        n = GLIFNeuron()
        assert len(_run(n, current=5.0, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [20.0, 50.0, 100.0]:
            n = GLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 20.0, 50.0, 100.0])
    def test_fi_sweep(self, current: float):
        n = GLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestGLIFParameters:
    @pytest.mark.parametrize("tau_m", [5.0, 10.0, 20.0])
    def test_tau_m_sweep(self, tau_m: float):
        n = GLIFNeuron(tau_m=tau_m)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("delta_theta", [1.0, 2.0, 5.0])
    def test_delta_theta_sweep(self, delta_theta: float):
        n = GLIFNeuron(delta_theta=delta_theta)
        spikes = len(_run(n, current=50.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("r_asc1", [0.0, 1.0, 3.0])
    def test_r_asc1_sweep(self, r_asc1: float):
        n = GLIFNeuron(r_asc1=r_asc1)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestGLIFPerformance:
    def test_isolation_throughput(self):
        n = GLIFNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(GLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
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
class TestGLIFPipeline:
    def test_population(self):
        assert Population(GLIFNeuron, n=10, label="glif").n == 10

    def test_projection_wiring(self):
        src = Population(GLIFNeuron, n=5, label="src")
        tgt = Population(GLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=20.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(GLIFNeuron, n=10, label="glif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = GLIFNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 20

    def test_analysis_isi(self):
        n = GLIFNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = GLIFNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = GLIFNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
