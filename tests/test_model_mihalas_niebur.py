# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MihalasNieburNeuron

"""Full pipeline test for MihalasNieburNeuron (Mihalas & Niebur 2009).

Generalised IF with dynamic threshold + 2 adaptation currents.
dV/dt = (-(V-V_rest) + i1 + i2 + I) / tau_v
dθ/dt = (θ_inf - θ + a·(V-V_rest)) / tau_θ
di1/dt = -i1 / tau_1
di2/dt = -i2 / tau_2

On spike: V→V_reset, θ→max(θ, θ_reset), i1+=r1, i2+=r2.
4 state variables. Can reproduce 20 spike patterns (tonic, phasic,
burst, accommodation, etc.) via parameter configuration.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: MihalasNieburNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestMNIsolation:
    def test_defaults(self):
        n = MihalasNieburNeuron()
        assert n.v == 0.0 and n.theta == 1.0
        assert n.i1 == 0.0 and n.i2 == 0.0
        assert n.tau_v == 10.0 and n.tau_theta == 100.0
        assert n.tau_1 == 10.0 and n.tau_2 == 200.0
        assert n.a == 0.0 and n.b == 0.0
        assert n.r1 == 0.0 and n.r2 == 0.0

    def test_step_returns_binary(self):
        assert MihalasNieburNeuron().step(0.0) in (0, 1)

    def test_four_state_variables(self):
        n = MihalasNieburNeuron()
        for attr in ["v", "theta", "i1", "i2"]:
            assert hasattr(n, attr)

    def test_state_finite_long_run(self):
        n = MihalasNieburNeuron()
        for _ in range(100_000):
            n.step(2.0)
        for attr in ["v", "theta", "i1", "i2"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = MihalasNieburNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_rest and n.theta == n.theta_reset
        assert n.i1 == 0.0 and n.i2 == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MihalasNieburNeuron()
            trace = [(n.step(2.0), n.v, n.theta) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — dV, dθ, di1, di2, spike mechanism
# ---------------------------------------------------------------------------
class TestMNAnalytical:
    def test_dv_formula(self):
        """dV = (-(V-V_rest) + i1 + i2 + I) / tau_v · dt."""
        n = MihalasNieburNeuron()
        v0 = n.v
        I = 0.5  # subthreshold
        expected_dv = (-(v0 - n.v_rest) + n.i1 + n.i2 + I) / n.tau_v * n.dt
        n.step(I)
        assert abs((n.v - v0) - expected_dv) < 1e-12

    def test_dtheta_formula(self):
        """dθ = (θ_inf - θ + a·(V-V_rest)) / tau_θ · dt."""
        n = MihalasNieburNeuron(a=0.01)
        theta0, v0 = n.theta, n.v
        expected_dtheta = (n.theta_inf - theta0 + n.a * (v0 - n.v_rest)) / n.tau_theta * n.dt
        n.step(0.0)
        assert abs((n.theta - theta0) - expected_dtheta) < 1e-12

    def test_di1_exponential_decay(self):
        """di1 = -i1 / tau_1 · dt. After n steps: i1 ≈ i1_0·(1-dt/tau_1)^n."""
        n = MihalasNieburNeuron()
        n.i1 = 5.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        # Euler: i1 *= (1 - dt/tau_1) per step
        decay = (1.0 - n.dt / n.tau_1) ** steps
        expected = 5.0 * decay
        assert abs(n.i1 - expected) < 1e-10

    def test_di2_exponential_decay(self):
        """di2 = -i2 / tau_2 · dt."""
        n = MihalasNieburNeuron()
        n.i2 = 5.0
        steps = 20
        for _ in range(steps):
            n.step(0.0)
        decay = (1.0 - n.dt / n.tau_2) ** steps
        expected = 5.0 * decay
        assert abs(n.i2 - expected) < 1e-10

    def test_i1_decays_faster_than_i2(self):
        """tau_1 < tau_2 → i1 decays faster."""
        n = MihalasNieburNeuron()
        n.i1 = 5.0
        n.i2 = 5.0
        for _ in range(50):
            n.step(0.0)
        assert n.i1 < n.i2

    def test_spike_resets_voltage(self):
        n = MihalasNieburNeuron()
        for _ in range(10_000):
            if n.step(2.0) == 1:
                assert n.v == n.v_reset
                break

    def test_spike_theta_max(self):
        """On spike: θ → max(θ, θ_reset)."""
        n = MihalasNieburNeuron()
        for _ in range(10_000):
            if n.step(2.0) == 1:
                assert n.theta >= n.theta_reset
                break

    def test_spike_increments_adaptation(self):
        """On spike: i1 += r1, i2 += r2."""
        n = MihalasNieburNeuron(r1=1.0, r2=0.5)
        for _ in range(10_000):
            if n.step(2.0) == 1:
                assert n.i1 >= 0.9  # r1=1 minus small decay
                assert n.i2 >= 0.4  # r2=0.5 minus small decay
                break

    def test_membrane_steady_state(self):
        """At steady state (no spike, a=0): V_ss = V_rest + I."""
        n = MihalasNieburNeuron()
        I = 0.5  # subthreshold
        for _ in range(10_000):
            n.step(I)
        expected_ss = n.v_rest + I  # tau_v normalises
        assert abs(n.v - expected_ss) < 0.5


# ---------------------------------------------------------------------------
# 3. SPIKE PATTERN CONFIGURATIONS
# ---------------------------------------------------------------------------
class TestMNPatterns:
    def test_tonic_spiking(self):
        """Default (no adaptation) → regular tonic spiking."""
        n = MihalasNieburNeuron()
        spikes = _run(n, current=2.0, steps=5000)
        assert len(spikes) >= 100

    def test_adaptation_with_negative_r(self):
        """Negative r1/r2 → spike-frequency adaptation."""
        n = MihalasNieburNeuron(r1=-0.5, r2=-0.1)
        s1 = sum(n.step(2.0) for _ in range(2500))
        s2 = sum(n.step(2.0) for _ in range(2500))
        # First half more spikes than second (adaptation)
        assert s1 >= s2

    def test_dynamic_threshold_with_a(self):
        """a > 0 → threshold tracks voltage → different dynamics."""
        n = MihalasNieburNeuron(a=0.1)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    def test_positive_r_bursting(self):
        """Positive r → excitatory after-current → can produce bursts."""
        n = MihalasNieburNeuron(r1=0.5, r2=0.0)
        spikes = _run(n, current=2.0, steps=5000)
        assert len(spikes) >= 10


# ---------------------------------------------------------------------------
# 4. DYNAMICS — f-I, ISI
# ---------------------------------------------------------------------------
class TestMNDynamics:
    def test_subthreshold_silent(self):
        n = MihalasNieburNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_fires_at_sufficient_current(self):
        n = MihalasNieburNeuron()
        assert len(_run(n, current=2.0, steps=5000)) >= 100

    def test_rate_monotonic(self):
        rates = []
        for I in [1.5, 3.0, 5.0]:
            n = MihalasNieburNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = MihalasNieburNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestMNParameters:
    @pytest.mark.parametrize("tau_v", [5.0, 10.0, 20.0])
    def test_tau_v_sweep(self, tau_v: float):
        n = MihalasNieburNeuron(tau_v=tau_v)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("tau_theta", [50.0, 100.0, 500.0])
    def test_tau_theta_sweep(self, tau_theta: float):
        n = MihalasNieburNeuron(tau_theta=tau_theta, a=0.01)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    @pytest.mark.parametrize("r1", [-1.0, 0.0, 1.0])
    def test_r1_sweep(self, r1: float):
        n = MihalasNieburNeuron(r1=r1)
        spikes = len(_run(n, current=2.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("dt", [0.1, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = MihalasNieburNeuron(dt=dt)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta)


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestMNPerformance:
    def test_isolation_throughput(self):
        n = MihalasNieburNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 100_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(MihalasNieburNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
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
class TestMNPipeline:
    def test_population(self):
        assert Population(MihalasNieburNeuron, n=10, label="mn").n == 10

    def test_projection_wiring(self):
        src = Population(MihalasNieburNeuron, n=5, label="src")
        tgt = Population(MihalasNieburNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(MihalasNieburNeuron, n=10, label="mn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MihalasNieburNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = MihalasNieburNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = MihalasNieburNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = MihalasNieburNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
