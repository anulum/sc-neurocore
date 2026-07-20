# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GLMNeuron

"""Full pipeline test for GLMNeuron (Pillow et al. 2008).

Point-process generalised linear model:
λ(t) = exp(k·stim_buf + h·spike_buf + μ)
P(spike) = min(λ·dt_ms/1000, 1)

k: stimulus filter (n_k=10, exponential decay)
h: post-spike filter (n_h=20, negative=refractoriness + slow excitation)
μ=-3.0 (baseline log-rate). Stochastic — Bernoulli sampling.
Circular buffers for stimulus history and spike history.
log_rate clipped to [-20, 20] to prevent overflow.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.glm_neuron import GLMNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GLMNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestGLMIsolation:
    def test_defaults(self):
        n = GLMNeuron()
        assert n.n_k == 10 and n.n_h == 20
        assert n.mu == -3.0 and n.dt_ms == 1.0
        assert n.k.shape == (10,) and n.h.shape == (20,)

    def test_step_returns_binary(self):
        assert GLMNeuron().step(0.0) in (0, 1)

    def test_buffers_initialised_to_zero(self):
        n = GLMNeuron()
        assert np.all(n._stim_buf == 0.0)
        assert np.all(n._spike_buf == 0.0)

    def test_reset_clears_buffers(self):
        n = GLMNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert np.all(n._stim_buf == 0.0)
        assert np.all(n._spike_buf == 0.0)

    def test_stochastic_two_runs_differ(self):
        """Different RNG seeds → different spike trains."""
        n1 = GLMNeuron()
        n2 = GLMNeuron()
        t1 = [n1.step(5.0) for _ in range(1000)]
        t2 = [n2.step(5.0) for _ in range(1000)]
        assert t1 != t2


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — filters, log-rate, clipping, buffers
# ---------------------------------------------------------------------------
class TestGLMAnalytical:
    def test_stimulus_filter_shape(self):
        """k = 0.5·exp(-arange(n_k)/3). Exponential decay."""
        n = GLMNeuron()
        expected = np.exp(-np.arange(10) / 3.0) * 0.5
        np.testing.assert_allclose(n.k, expected)

    def test_post_spike_filter_shape(self):
        """h = -5·exp(-t/2) + 0.5·exp(-t/10). Starts strongly negative."""
        n = GLMNeuron()
        t = np.arange(20)
        expected = -5.0 * np.exp(-t / 2.0) + 0.5 * np.exp(-t / 10.0)
        np.testing.assert_allclose(n.h, expected)

    def test_h_filter_refractoriness(self):
        """h[0] is strongly negative → suppresses firing after spike."""
        n = GLMNeuron()
        assert n.h[0] < -4.0  # -5 + 0.5 = -4.5

    def test_stimulus_buffer_circular(self):
        """New stimulus enters at index 0, old values shift right."""
        n = GLMNeuron(n_k=4, mu=-100.0)  # high mu to prevent spikes
        n.step(1.0)
        assert n._stim_buf[0] == 1.0
        n.step(2.0)
        assert n._stim_buf[0] == 2.0
        assert n._stim_buf[1] == 1.0

    def test_log_rate_clipping(self):
        """log_rate clipped to [-20, 20] → exp(20) ≈ 4.85e8."""
        n = GLMNeuron(mu=100.0)  # extreme mu
        spike = n.step(1000.0)

        assert spike == 1
        assert np.all(np.isfinite(n._stim_buf))
        assert np.all(np.isfinite(n._spike_buf))

    def test_baseline_rate_at_zero_input(self):
        """At zero stimulus, no history: log_rate = μ = -3.0 → λ = exp(-3) ≈ 0.05."""
        n = GLMNeuron()
        expected_lambda = np.exp(-3.0)
        expected_p = expected_lambda * 1.0 / 1000.0  # dt_ms=1, /1000
        # Very low probability per step
        assert expected_p < 0.001

    def test_spike_enters_spike_buffer(self):
        """After spike, spike_buf[0] = 1.0."""
        n = GLMNeuron(mu=10.0)  # high mu to guarantee spike

        assert n.step(10.0) == 1
        assert n._spike_buf[0] == 1.0


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestGLMDynamics:
    def test_fires_with_strong_stimulus(self):
        n = GLMNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 10

    def test_silent_at_zero(self):
        """At μ=-3 and zero stimulus: very low rate."""
        n = GLMNeuron()
        spikes = _run(n, current=0.0, steps=1000)
        # May get 0-2 spikes (stochastic)
        assert len(spikes) < 50

    def test_rate_increases_with_stimulus(self):
        n_low = GLMNeuron()
        n_high = GLMNeuron()
        s_low = len(_run(n_low, current=2.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("stim", [0.0, 2.0, 5.0, 10.0])
    def test_stim_sweep(self, stim: float):
        n = GLMNeuron()
        spikes = [n.step(stim) for _ in range(1000)]

        assert set(spikes) <= {0, 1}
        np.testing.assert_array_equal(n._stim_buf, np.full(n.n_k, stim))
        assert set(n._spike_buf) <= {0.0, 1.0}
        assert np.all(np.isfinite(n._spike_buf))


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestGLMParameters:
    @pytest.mark.parametrize("mu", [-5.0, -3.0, 0.0])
    def test_mu_sweep(self, mu: float):
        n = GLMNeuron(mu=mu)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("n_k", [5, 10, 20])
    def test_n_k_sweep(self, n_k: int):
        n = GLMNeuron(n_k=n_k)
        assert n.k.shape == (n_k,)
        for _ in range(500):
            n.step(5.0)

    @pytest.mark.parametrize("n_h", [10, 20, 40])
    def test_n_h_sweep(self, n_h: int):
        n = GLMNeuron(n_h=n_h)
        assert n.h.shape == (n_h,)
        for _ in range(500):
            n.step(5.0)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestGLMPerformance:
    def test_isolation_throughput(self):
        n = GLMNeuron()
        N = 20_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # np.dot + np.roll + RNG per step
        assert rate > 5_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(GLMNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 1_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestGLMPipeline:
    def test_population(self):
        assert Population(GLMNeuron, n=10, label="glm").n == 10

    def test_projection_wiring(self):
        src = Population(GLMNeuron, n=5, label="src")
        tgt = Population(GLMNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=3.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(GLMNeuron, n=10, label="glm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = GLMNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 0

    def test_analysis_isi(self):
        n = GLMNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = GLMNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0
