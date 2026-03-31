# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: CourageNekorkinMapNeuron

"""Full pipeline test for CourageNekorkinMapNeuron (Courbage et al. 2007).

Piecewise-linear Lorenz-type spiking map:
x_{n+1} = f(x_n) + y_n + I + j
y_{n+1} = y_n - β·(x_n + 1)

f(x) = α·x           if x < 0
     = α·x/(1+α·x)   if x ≥ 0  (saturating)

α=3.0 (expansion), β=0.001 (slow y), j=0.1 (bias).
Clipped to ±1e6. Spike on upward x_threshold crossing.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: CourageNekorkinMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestCNIsolation:
    def test_defaults(self):
        n = CourageNekorkinMapNeuron()
        assert n.x == 0.0 and n.y == 0.0
        assert n.alpha == 3.0 and n.beta == 0.001 and n.j == 0.1
        assert n.x_threshold == 1.0

    def test_step_returns_binary(self):
        assert CourageNekorkinMapNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(50_000):
            n.step(0.5)
        assert np.isfinite(n.x) and np.isfinite(n.y)

    def test_reset_restores_defaults(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(1000):
            n.step(0.5)
        n.reset()
        assert n.x == 0.0 and n.y == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = CourageNekorkinMapNeuron()
            trace = [(n.step(0.5), n.x, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — piecewise f(x), update formulas, clipping
# ---------------------------------------------------------------------------
class TestCNAnalytical:
    def test_f_negative_branch(self):
        """x < 0: f(x) = α·x (linear expansion)."""
        n = CourageNekorkinMapNeuron()
        assert abs(n._f(-1.0) - (-3.0)) < 1e-12
        assert abs(n._f(-0.5) - (-1.5)) < 1e-12

    def test_f_positive_branch(self):
        """x ≥ 0: f(x) = α·x/(1+α·x) (saturating)."""
        n = CourageNekorkinMapNeuron()
        # f(0) = 0
        assert abs(n._f(0.0)) < 1e-12
        # f(1) = 3/(1+3) = 0.75
        assert abs(n._f(1.0) - 0.75) < 1e-12
        # f(∞) → 1.0 (saturation)

    def test_f_continuity_at_zero(self):
        """f(0⁻) = 0, f(0) = 0. Continuous."""
        n = CourageNekorkinMapNeuron()
        assert abs(n._f(-1e-10) - n._f(0.0)) < 1e-6

    def test_f_saturation(self):
        """Positive branch saturates: f(x) → 1 as x → ∞."""
        n = CourageNekorkinMapNeuron()
        assert n._f(1000.0) > 0.99

    def test_x_update_formula(self):
        """x_new = f(x) + y + I + j, clipped to ±1e6."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y
        I = 0.3
        expected_x = n._f(x0) + y0 + I + n.j
        n.step(I)
        assert abs(n.x - expected_x) < 1e-10

    def test_y_update_formula(self):
        """y_new = y - β·(x+1)."""
        n = CourageNekorkinMapNeuron()
        x0, y0 = n.x, n.y
        expected_dy = -n.beta * (x0 + 1.0)
        n.step(0.0)
        assert abs((n.y - y0) - expected_dy) < 1e-14

    def test_beta_slow_timescale(self):
        """β=0.001 → y changes very slowly."""
        n = CourageNekorkinMapNeuron()
        y0 = n.y
        n.step(0.0)
        assert abs(n.y - y0) < 0.01

    def test_clipping_prevents_divergence(self):
        """Values clipped to ±1e6."""
        n = CourageNekorkinMapNeuron()
        n.x = 1e7
        n.step(0.0)
        assert abs(n.x) <= 1e6 + 1


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestCNDynamics:
    def test_fires_with_input(self):
        n = CourageNekorkinMapNeuron()
        spikes = _run(n, current=0.5, steps=5000)
        assert len(spikes) >= 1

    def test_rate_increases_with_input(self):
        s_low = len(_run(CourageNekorkinMapNeuron(), 0.1, 5000))
        s_high = len(_run(CourageNekorkinMapNeuron(), 1.0, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_fi_sweep(self, current: float):
        n = CourageNekorkinMapNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.x)

    def test_upward_crossing_only(self):
        n = CourageNekorkinMapNeuron()
        prev_x = n.x
        for _ in range(5000):
            spike = n.step(0.5)
            if spike == 1:
                assert prev_x < n.x_threshold
            prev_x = n.x


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestCNParameters:
    @pytest.mark.parametrize("alpha", [2.0, 3.0, 5.0])
    def test_alpha_sweep(self, alpha: float):
        n = CourageNekorkinMapNeuron(alpha=alpha)
        for _ in range(5000):
            n.step(0.5)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("beta", [0.0005, 0.001, 0.005])
    def test_beta_sweep(self, beta: float):
        n = CourageNekorkinMapNeuron(beta=beta)
        for _ in range(5000):
            n.step(0.5)
        assert np.isfinite(n.y)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestCNPerformance:
    def test_isolation_throughput(self):
        n = CourageNekorkinMapNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 100_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(CourageNekorkinMapNeuron, n=20, label="bench")
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
class TestCNPipeline:
    def test_population(self):
        assert Population(CourageNekorkinMapNeuron, n=10, label="cn").n == 10

    def test_projection_wiring(self):
        src = Population(CourageNekorkinMapNeuron, n=5, label="src")
        tgt = Population(CourageNekorkinMapNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(CourageNekorkinMapNeuron, n=10, label="cn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 0

    def test_analysis_isi(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = CourageNekorkinMapNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0
