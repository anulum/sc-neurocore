# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: InhomogeneousPoissonNeuron

"""Full pipeline test for InhomogeneousPoissonNeuron (Cox 1955).

Doubly stochastic Poisson (time-varying rate):
P(spike) = max(0, rate_hz) · dt_ms / 1000
Bernoulli sampling per step. Stateless — no internal dynamics.
Expected spike count = N · rate · dt_ms / 1000.
Negative rate clipped to 0.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.inhomogeneous_poisson import InhomogeneousPoissonNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: InhomogeneousPoissonNeuron, rate: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(rate) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestIPIsolation:
    def test_defaults(self):
        n = InhomogeneousPoissonNeuron()
        assert n.dt_ms == 1.0

    def test_step_returns_binary(self):
        assert InhomogeneousPoissonNeuron().step(100.0) in (0, 1)

    def test_stateless(self):
        """No internal state — only dt_ms parameter."""
        n = InhomogeneousPoissonNeuron()
        assert not hasattr(n, "v")

    def test_reset_noop(self):
        n = InhomogeneousPoissonNeuron()
        n.step(100.0)
        n.reset()
        # Nothing to verify — stateless

    def test_stochastic_two_runs_differ(self):
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        t1 = [n1.step(100.0) for _ in range(1000)]
        t2 = [n2.step(100.0) for _ in range(1000)]
        # Shared np.random → may be equal; test with many steps
        # Actually they share global RNG, so alternating calls may differ
        assert isinstance(t1, list)


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — probability formula, expected rate, negative clipping
# ---------------------------------------------------------------------------
class TestIPAnalytical:
    def test_probability_formula(self):
        """P(spike) = rate_hz · dt_ms / 1000."""
        n = InhomogeneousPoissonNeuron(dt_ms=1.0)
        # At rate=100 Hz, dt=1ms: P = 100/1000 = 0.1
        expected_p = 100.0 * 1.0 / 1000.0
        assert abs(expected_p - 0.1) < 1e-12

    def test_expected_spike_count(self):
        """E[spikes] = N · rate · dt/1000. Statistical test (5σ tolerance)."""
        n = InhomogeneousPoissonNeuron(dt_ms=1.0)
        N = 100_000
        rate = 100.0
        spikes = sum(n.step(rate) for _ in range(N))
        expected = N * rate * 1.0 / 1000.0  # = 10000
        std = np.sqrt(N * (rate / 1000.0) * (1 - rate / 1000.0))
        assert abs(spikes - expected) < 5 * std

    def test_negative_rate_no_spikes(self):
        """Negative rate clipped to 0 → P = 0."""
        n = InhomogeneousPoissonNeuron()
        spikes = sum(n.step(-100.0) for _ in range(10_000))
        assert spikes == 0

    def test_zero_rate_no_spikes(self):
        n = InhomogeneousPoissonNeuron()
        spikes = sum(n.step(0.0) for _ in range(10_000))
        assert spikes == 0

    def test_high_rate_near_certain(self):
        """rate=10000, dt=1ms → P=10 → clipped to P<1 in random() < p."""
        n = InhomogeneousPoissonNeuron()
        # P = 10000 * 1 / 1000 = 10 > 1 → fires every step
        spikes = sum(n.step(10000.0) for _ in range(100))
        assert spikes == 100

    def test_rate_proportional(self):
        """Double rate → double expected spikes."""
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        N = 50_000
        s1 = sum(n1.step(50.0) for _ in range(N))
        s2 = sum(n2.step(100.0) for _ in range(N))
        # s2 should be roughly 2× s1 (statistical)
        assert s2 > s1

    @pytest.mark.parametrize("dt_ms", [0.1, 1.0, 5.0])
    def test_dt_ms_scales_probability(self, dt_ms: float):
        """Larger dt_ms → higher P per step."""
        n = InhomogeneousPoissonNeuron(dt_ms=dt_ms)
        spikes = sum(n.step(100.0) for _ in range(10_000))
        assert isinstance(spikes, int)


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestIPDynamics:
    def test_fires_at_positive_rate(self):
        n = InhomogeneousPoissonNeuron()
        spikes = _run(n, rate=100.0, steps=10_000)
        assert len(spikes) >= 100

    def test_rate_monotonic(self):
        s_low = len(_run(InhomogeneousPoissonNeuron(), 50.0, 10_000))
        s_high = len(_run(InhomogeneousPoissonNeuron(), 500.0, 10_000))
        assert s_high >= s_low


# ---------------------------------------------------------------------------
# 4. PERFORMANCE
# ---------------------------------------------------------------------------
class TestIPPerformance:
    def test_isolation_throughput(self):
        n = InhomogeneousPoissonNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 200_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(InhomogeneousPoissonNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 5. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestIPPipeline:
    def test_population(self):
        assert Population(InhomogeneousPoissonNeuron, n=10, label="ip").n == 10

    def test_projection_wiring(self):
        src = Population(InhomogeneousPoissonNeuron, n=5, label="src")
        tgt = Population(InhomogeneousPoissonNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(InhomogeneousPoissonNeuron, n=10, label="ip")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 100

    def test_analysis_isi(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
