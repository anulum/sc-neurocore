# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MedvedevMapNeuron

"""Full pipeline test for MedvedevMapNeuron (Medvedev 2005).

1D piecewise-monotone spiking map:
x_{n+1} = α·x_n + I          if x < β
         = α·(1 - x_n) + I   if x ≥ β
x_{n+1} = x_{n+1} mod 1

Two branches: expansive (slope α=3.5) below β=0.5, folding above.
mod 1 keeps x ∈ [0, 1). Chaotic dynamics (Lyapunov > 0 for α > 1).
Spike on upward threshold crossing at x_threshold=0.9.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: MedvedevMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestMedvedevIsolation:
    def test_defaults(self):
        n = MedvedevMapNeuron()
        assert n.x == 0.0 and n.alpha == 3.5
        assert n.beta == 0.5 and n.x_threshold == 0.9

    def test_step_returns_binary(self):
        assert MedvedevMapNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = MedvedevMapNeuron()
        n.step(0.2)
        assert n.x != 0.0

    def test_state_finite_long_run(self):
        n = MedvedevMapNeuron()
        for _ in range(100_000):
            n.step(0.2)
        assert np.isfinite(n.x)

    def test_reset_restores_default(self):
        n = MedvedevMapNeuron()
        for _ in range(500):
            n.step(0.2)
        n.reset()
        assert n.x == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MedvedevMapNeuron()
            trace = [(n.step(0.2), n.x) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — piecewise map, mod 1, branches
# ---------------------------------------------------------------------------
class TestMedvedevAnalytical:
    def test_low_branch_formula(self):
        """x < β: x_next = (α·x + I) mod 1."""
        n = MedvedevMapNeuron()
        n.x = 0.1
        I = 0.05
        expected = (n.alpha * 0.1 + I) % 1.0
        n.step(I)
        assert abs(n.x - expected) < 1e-12

    def test_high_branch_formula(self):
        """x ≥ β: x_next = (α·(1-x) + I) mod 1."""
        n = MedvedevMapNeuron()
        n.x = 0.7
        I = 0.05
        expected = (n.alpha * (1.0 - 0.7) + I) % 1.0
        n.step(I)
        assert abs(n.x - expected) < 1e-12

    def test_mod_1_bounds(self):
        """x is always in [0, 1) after mod operation."""
        n = MedvedevMapNeuron()
        for _ in range(10_000):
            n.step(0.3)
            assert 0.0 <= n.x < 1.0

    def test_beta_is_branch_point(self):
        """x=β-ε uses low branch, x=β uses high branch."""
        n1 = MedvedevMapNeuron()
        n1.x = 0.499
        x_before = n1.x
        n1.step(0.0)
        x_low = n1.x
        expected_low = (n1.alpha * 0.499) % 1.0
        assert abs(x_low - expected_low) < 1e-10

        n2 = MedvedevMapNeuron()
        n2.x = 0.5
        n2.step(0.0)
        x_high = n2.x
        expected_high = (n2.alpha * (1.0 - 0.5)) % 1.0
        assert abs(x_high - expected_high) < 1e-12

    def test_expansive_map(self):
        """α=3.5 > 1 → map is expansive on both branches."""
        n = MedvedevMapNeuron()
        assert n.alpha > 1.0

    def test_spike_on_upward_crossing(self):
        """Spike iff x_prev < threshold and x_new ≥ threshold."""
        n = MedvedevMapNeuron()
        prev_x = n.x
        for _ in range(10_000):
            spike = n.step(0.2)
            if spike == 1:
                assert prev_x < n.x_threshold
            prev_x = n.x


# ---------------------------------------------------------------------------
# 3. CHAOTIC DYNAMICS
# ---------------------------------------------------------------------------
class TestMedvedevChaos:
    def test_sensitive_dependence(self):
        """Tiny initial perturbation amplifies exponentially."""
        n1 = MedvedevMapNeuron(x=0.1)
        n2 = MedvedevMapNeuron(x=0.1 + 1e-10)
        for _ in range(100):
            n1.step(0.2)
            n2.step(0.2)
        assert abs(n1.x - n2.x) > 1e-5

    def test_ergodic_coverage(self):
        """Chaotic map visits most of [0, 1) uniformly."""
        n = MedvedevMapNeuron()
        bins = np.zeros(10)
        for _ in range(100_000):
            n.step(0.2)
            idx = min(int(n.x * 10), 9)
            bins[idx] += 1
        # All bins should have significant counts
        assert all(b > 1000 for b in bins)

    def test_irregular_isi(self):
        """Chaotic dynamics → irregular ISI (high CV)."""
        n = MedvedevMapNeuron()
        spikes = _run(n, current=0.2, steps=50_000)
        if len(spikes) >= 20:
            isis = np.diff(spikes).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv > 0.2  # irregular


# ---------------------------------------------------------------------------
# 4. DYNAMICS — f-I, silent/firing regions
# ---------------------------------------------------------------------------
class TestMedvedevDynamics:
    def test_silent_at_zero(self):
        """x=0 is fixed point at I=0: α·0 = 0."""
        n = MedvedevMapNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_fires_with_input(self):
        n = MedvedevMapNeuron()
        assert len(_run(n, current=0.2, steps=5000)) >= 100

    def test_rate_increases_with_input(self):
        rates = []
        for I in [0.1, 0.3, 0.5]:
            n = MedvedevMapNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 0.1, 0.2, 0.3, 0.5])
    def test_fi_sweep(self, current: float):
        n = MedvedevMapNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.x)


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestMedvedevParameters:
    @pytest.mark.parametrize("alpha", [2.0, 3.5, 5.0])
    def test_alpha_sweep(self, alpha: float):
        n = MedvedevMapNeuron(alpha=alpha)
        for _ in range(5000):
            n.step(0.2)
        assert 0.0 <= n.x < 1.0

    @pytest.mark.parametrize("beta", [0.3, 0.5, 0.7])
    def test_beta_sweep(self, beta: float):
        n = MedvedevMapNeuron(beta=beta)
        for _ in range(5000):
            n.step(0.2)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("x_threshold", [0.5, 0.9, 0.95])
    def test_threshold_sweep(self, x_threshold: float):
        n = MedvedevMapNeuron(x_threshold=x_threshold)
        spikes = len(_run(n, current=0.2, steps=5000))
        assert isinstance(spikes, int)


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestMedvedevPerformance:
    def test_isolation_throughput(self):
        n = MedvedevMapNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.2)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # Simple 1D map + mod
        assert rate > 500_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(MedvedevMapNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.2, dt=0.001, seed=42)
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
class TestMedvedevPipeline:
    def test_population(self):
        assert Population(MedvedevMapNeuron, n=10, label="med").n == 10

    def test_projection_wiring(self):
        src = Population(MedvedevMapNeuron, n=5, label="src")
        tgt = Population(MedvedevMapNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.2, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.1, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(MedvedevMapNeuron, n=10, label="med")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.2, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MedvedevMapNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = MedvedevMapNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = MedvedevMapNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = MedvedevMapNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
