# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: KLIFNeuron

"""Full pipeline test for KLIFNeuron (k-LIF, SNN backprop variant).

LIF with learnable scaling factor k:
V[t+1] = α·V[t] + k·I
α = exp(-dt/τ), precomputed. k=1.0 (trainable).
Spike: V → V_reset when V ≥ threshold.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.klif import KLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: KLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestKLIFIsolation:
    def test_defaults(self):
        n = KLIFNeuron()
        assert n.v == 0.0 and n.k == 1.0
        assert n.tau == 10.0 and n.v_threshold == 1.0

    def test_precomputed_alpha(self):
        n = KLIFNeuron()
        assert abs(n.alpha - np.exp(-1.0 / 10.0)) < 1e-14

    def test_step_returns_binary(self):
        assert KLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = KLIFNeuron()
        for _ in range(100_000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset_restores_default(self):
        n = KLIFNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = KLIFNeuron()
            trace = [(n.step(1.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — V formula, k scaling, alpha decay
# ---------------------------------------------------------------------------
class TestKLIFAnalytical:
    def test_v_update_formula(self):
        """V = α·V + k·I."""
        n = KLIFNeuron()
        v0 = n.v
        I = 0.5
        expected = n.alpha * v0 + n.k * I
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-12

    def test_k_scales_input(self):
        """k=2 → double effective input."""
        n1 = KLIFNeuron(k=1.0, v_threshold=100.0)
        n2 = KLIFNeuron(k=2.0, v_threshold=100.0)
        for _ in range(100):
            n1.step(1.0)
            n2.step(1.0)
        assert n2.v > n1.v

    def test_alpha_decay_without_input(self):
        """V decays by α per step when I=0."""
        n = KLIFNeuron(v_threshold=100.0)
        n.v = 0.5
        for _ in range(10):
            n.step(0.0)
        expected = 0.5 * n.alpha**10
        assert abs(n.v - expected) < 1e-10

    def test_spike_resets_voltage(self):
        n = KLIFNeuron()
        for _ in range(10_000):
            if n.step(1.0) == 1:
                assert n.v == n.v_reset
                break

    def test_zero_k_no_integration(self):
        """k=0 → V never integrates input."""
        n = KLIFNeuron(k=0.0)
        for _ in range(1000):
            n.step(10.0)
        # V = alpha^n * 0 = 0 (starts at 0, no input scaled)
        assert abs(n.v) < 1e-10


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestKLIFDynamics:
    def test_fires_under_drive(self):
        n = KLIFNeuron()
        assert len(_run(n, current=1.0, steps=5000)) >= 100

    def test_subthreshold_silent(self):
        n = KLIFNeuron()
        assert len(_run(n, current=0.01, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [0.5, 1.0, 5.0]:
            n = KLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_fi_sweep(self, current: float):
        n = KLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestKLIFParameters:
    @pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
    def test_k_sweep(self, k: float):
        n = KLIFNeuron(k=k)
        spikes = len(_run(n, current=1.0, steps=5000))
        assert isinstance(spikes, int)

    def test_higher_k_more_spikes(self):
        s_low = len(_run(KLIFNeuron(k=0.5), 1.0, 5000))
        s_high = len(_run(KLIFNeuron(k=2.0), 1.0, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("tau", [5.0, 10.0, 20.0])
    def test_tau_sweep(self, tau: float):
        n = KLIFNeuron(tau=tau)
        for _ in range(5000):
            n.step(1.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestKLIFPerformance:
    def test_isolation_throughput(self):
        n = KLIFNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 500_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(KLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
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
class TestKLIFPipeline:
    def test_population(self):
        assert Population(KLIFNeuron, n=10, label="klif").n == 10

    def test_projection_wiring(self):
        src = Population(KLIFNeuron, n=5, label="src")
        tgt = Population(KLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(KLIFNeuron, n=10, label="klif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = KLIFNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = KLIFNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = KLIFNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
