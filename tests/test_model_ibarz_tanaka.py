# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: IbarzTanakaMapNeuron

"""Full pipeline test for IbarzTanakaMapNeuron (Ibarz et al. 2007).

Piecewise-linear bursting map: f(x) = α/(1-x) for x≤0, α+βx otherwise.
Slow y variable modulates bursting via µ."""

from __future__ import annotations

import numpy as np

import time

import pytest

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


class TestITMapIsolation:
    def test_construction(self):
        n = IbarzTanakaMapNeuron()
        assert n.x == -1.0
        assert n.y == -2.5

    def test_step_returns_binary(self):
        assert IbarzTanakaMapNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = IbarzTanakaMapNeuron()
        assert sum(n.step(0.5) for _ in range(5000)) == 0

    def test_spikes_under_drive(self):
        n = IbarzTanakaMapNeuron()
        assert sum(n.step(2.0) for _ in range(10000)) > 50

    def test_piecewise_f(self):
        """f(x) should switch at x=0."""
        n = IbarzTanakaMapNeuron()
        f_neg = n._f(-1.0)
        f_pos = n._f(1.0)
        assert abs(f_neg - 3.65 / 2.0) < 1e-10
        assert abs(f_pos - (3.65 + 0.25)) < 1e-10

    def test_slow_y_dynamics(self):
        """y changes slowly (µ=0.0005)."""
        n = IbarzTanakaMapNeuron()
        y0 = n.y
        for _ in range(1000):
            n.step(2.0)
        assert n.y != y0

    def test_reset_on_spike(self):
        """x should reset to x_reset when threshold crossed."""
        n = IbarzTanakaMapNeuron()
        for _ in range(10000):
            if n.step(2.0):
                assert n.x == n.x_reset
                break

    def test_rate_increases_with_input(self):
        n_low = IbarzTanakaMapNeuron()
        n_high = IbarzTanakaMapNeuron()
        s_low = sum(n_low.step(1.5) for _ in range(10000))
        s_high = sum(n_high.step(3.0) for _ in range(10000))
        assert s_high > s_low

    def test_numerical_stability(self):
        for I in [0.0, 1.0, 2.0, 3.0]:
            n = IbarzTanakaMapNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.x), f"x NaN at I={I}"
            assert np.isfinite(n.y), f"y NaN at I={I}"

    def test_reset(self):
        n = IbarzTanakaMapNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.x == -1.0
        assert n.y == -2.5

    def test_deterministic(self):
        n1 = IbarzTanakaMapNeuron()
        n2 = IbarzTanakaMapNeuron()
        for _ in range(500):
            assert n1.step(2.0) == n2.step(2.0)


class TestITMapAnalytical:
    def test_f_negative_branch(self):
        """x ≤ 0: f(x) = α/(1-x)."""
        n = IbarzTanakaMapNeuron()
        assert abs(n._f(-1.0) - 3.65 / 2.0) < 1e-12
        assert abs(n._f(0.0) - 3.65) < 1e-12

    def test_f_positive_branch(self):
        """x > 0: f(x) = α + β·x."""
        n = IbarzTanakaMapNeuron()
        assert abs(n._f(1.0) - (3.65 + 0.25)) < 1e-12
        assert abs(n._f(2.0) - (3.65 + 0.50)) < 1e-12

    def test_f_continuity_at_zero(self):
        """f(0⁻) = α/1 = α, f(0⁺) = α + β·0 = α. Continuous."""
        n = IbarzTanakaMapNeuron()
        assert abs(n._f(0.0) - n._f(1e-10)) < 0.01

    def test_x_update_formula(self):
        """x_new = f(x) + y + I."""
        n = IbarzTanakaMapNeuron()
        x0, y0 = n.x, n.y
        I = 1.0
        expected_x = n._f(x0) + y0 + I
        expected_y = y0 - n.mu * (x0 + 1.0) + n.mu * n.sigma
        n.step(I)
        if n.x != n.x_reset:  # no spike
            assert abs(n.x - expected_x) < 1e-12
        assert abs(n.y - expected_y) < 1e-14

    def test_y_update_formula(self):
        """y_new = y - µ·(x+1) + µ·σ."""
        n = IbarzTanakaMapNeuron()
        x0, y0 = n.x, n.y
        expected_dy = -n.mu * (x0 + 1.0) + n.mu * n.sigma
        n.step(0.0)
        assert abs((n.y - y0) - expected_dy) < 1e-14

    def test_mu_slow_timescale(self):
        """µ=0.0005 → y changes very slowly."""
        n = IbarzTanakaMapNeuron()
        y0 = n.y
        n.step(0.0)
        assert abs(n.y - y0) < 0.01

    @pytest.mark.parametrize("current", [0.0, 1.0, 2.0, 3.0, 5.0])
    def test_fi_sweep(self, current: float):
        n = IbarzTanakaMapNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.x) and np.isfinite(n.y)


class TestITMapPerformance:
    def test_isolation_throughput(self):
        n = IbarzTanakaMapNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 200_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(IbarzTanakaMapNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


class TestITMapPipeline:
    def test_population(self):
        assert Population(IbarzTanakaMapNeuron, n=10, label="itm").n == 10

    def test_projection_wiring(self):
        src = Population(IbarzTanakaMapNeuron, n=5, label="src")
        tgt = Population(IbarzTanakaMapNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(IbarzTanakaMapNeuron, n=10, label="itm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = IbarzTanakaMapNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(10_000)])
        assert spike_count(train) > 50

    def test_analysis_isi(self):
        n = IbarzTanakaMapNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(10_000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = IbarzTanakaMapNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = IbarzTanakaMapNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(10_000)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
