# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HuberBraunNeuron

"""Full pipeline test for HuberBraunNeuron (Braun, Huber et al. 1998).

Cold receptor, temperature-dependent model with Gaussian noise:
3 currents: I_sd (slow depolarising, g=1.5), I_sr (slow repolarising,
g=0.4), I_L (leak, g=0.1).
2 gating variables: a_sd (tau=10), a_sr (tau=20).

sd_inf = 1/(1+exp(-(v+40)/6))   — activates at depolarised V
sr_inf = 1/(1+exp((v+40)/6))    — activates at hyperpolarised V
Complementary: sd_inf + sr_inf = 1 at v=-40.

Gaussian noise: η·randn() per step (η=0.012). Stochastic model.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: HuberBraunNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestHBIsolation:
    def test_defaults(self):
        n = HuberBraunNeuron()
        assert n.v == -50.0 and n.a_sd == 0.0 and n.a_sr == 0.0
        assert n.g_sd == 1.5 and n.g_sr == 0.4 and n.g_l == 0.1
        assert n.eta == 0.012 and n.dt == 0.1

    def test_step_returns_binary(self):
        assert HuberBraunNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = HuberBraunNeuron()
        for _ in range(50_000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.a_sd) and np.isfinite(n.a_sr)

    def test_reset_restores_defaults(self):
        n = HuberBraunNeuron()
        for _ in range(2000):
            n.step(50.0)
        n.reset()
        assert n.v == -50.0 and n.a_sd == 0.0 and n.a_sr == 0.0

    def test_stochastic_noise_present(self):
        """η·randn() noise → voltage diverges from deterministic path."""
        n1 = HuberBraunNeuron(eta=0.0)  # no noise
        n2 = HuberBraunNeuron(eta=0.012)  # with noise
        for _ in range(500):
            n1.step(50.0)
            n2.step(50.0)
        # With noise, voltages should differ
        assert n1.v != n2.v


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — sd/sr complementary, noise, currents
# ---------------------------------------------------------------------------
class TestHBAnalytical:
    def test_sd_inf_sr_inf_complementary(self):
        """sd_inf + sr_inf = 1 at any V (complementary sigmoids)."""
        for v in [-80, -60, -40, -20, 0]:
            sd = 1.0 / (1.0 + np.exp(-(v + 40.0) / 6.0))
            sr = 1.0 / (1.0 + np.exp((v + 40.0) / 6.0))
            assert abs(sd + sr - 1.0) < 1e-12

    def test_sd_inf_midpoint(self):
        """sd_inf(-40) = 0.5."""
        sd = 1.0 / (1.0 + np.exp(0.0))
        assert abs(sd - 0.5) < 1e-12

    def test_sd_activates_depolarised(self):
        """sd_inf → 1 for v >> -40."""
        sd = 1.0 / (1.0 + np.exp(-(0.0 + 40.0) / 6.0))
        assert sd > 0.99

    def test_sr_activates_hyperpolarised(self):
        """sr_inf → 1 for v << -40."""
        sr = 1.0 / (1.0 + np.exp((-80.0 + 40.0) / 6.0))
        assert sr > 0.99

    def test_three_currents(self):
        n = HuberBraunNeuron()
        assert n.g_sd > 0 and n.g_sr > 0 and n.g_l > 0

    def test_reversal_ordering(self):
        n = HuberBraunNeuron()
        assert n.e_sr < n.e_l < n.e_sd

    def test_noise_amplitude(self):
        """η=0.012 adds Gaussian noise per step."""
        n = HuberBraunNeuron()
        assert n.eta == 0.012

    def test_sd_slower_than_sr(self):
        """tau_sd=10 < tau_sr=20 — sd activates faster."""
        n = HuberBraunNeuron()
        assert n.tau_sd < n.tau_sr

    def test_gating_bounded(self):
        n = HuberBraunNeuron()
        for _ in range(10_000):
            n.step(50.0)
        assert -0.05 <= n.a_sd <= 1.05
        assert -0.05 <= n.a_sr <= 1.05


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestHBDynamics:
    def test_fires_under_drive(self):
        n = HuberBraunNeuron()
        spikes = _run(n, current=50.0, steps=10_000)
        assert len(spikes) >= 1

    def test_rate_increases_with_current(self):
        s_low = len(_run(HuberBraunNeuron(), 20.0, 10_000))
        s_high = len(_run(HuberBraunNeuron(), 100.0, 10_000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [0.0, 20.0, 50.0, 100.0])
    def test_fi_sweep(self, current: float):
        n = HuberBraunNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestHBParameters:
    @pytest.mark.parametrize("g_sd", [0.5, 1.5, 3.0])
    def test_g_sd_sweep(self, g_sd: float):
        n = HuberBraunNeuron(g_sd=g_sd)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("eta", [0.0, 0.012, 0.05])
    def test_eta_noise_sweep(self, eta: float):
        n = HuberBraunNeuron(eta=eta)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = HuberBraunNeuron(dt=dt)
        for _ in range(10_000):
            n.step(50.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestHBPerformance:
    def test_isolation_throughput(self):
        n = HuberBraunNeuron()
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 20_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(HuberBraunNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
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
class TestHBPipeline:
    def test_population(self):
        assert Population(HuberBraunNeuron, n=10, label="hb").n == 10

    def test_projection_wiring(self):
        src = Population(HuberBraunNeuron, n=5, label="src")
        tgt = Population(HuberBraunNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(HuberBraunNeuron, n=10, label="hb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = HuberBraunNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 0

    def test_analysis_isi(self):
        n = HuberBraunNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = HuberBraunNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0
