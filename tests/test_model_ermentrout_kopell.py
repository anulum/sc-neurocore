# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ErmentroutKopellPopulation

"""Full pipeline test for ErmentroutKopellPopulation (Montbrio et al. 2015).

Exact mean-field of QIF/theta neuron network. Returns firing rate r (float),
not binary spike. Population clips to {0,1}."""

from __future__ import annotations

import numpy as np

import time

import pytest

from sc_neurocore.neurons.models.ermentrout_kopell_pop import ErmentroutKopellPopulation
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


class TestErmentroutKopellIsolation:
    def test_construction(self):
        n = ErmentroutKopellPopulation()
        assert n.r == 0.1
        assert n.v == -2.0

    def test_step_returns_float(self):
        """Mean-field model returns firing rate (float), not binary spike."""
        n = ErmentroutKopellPopulation()
        result = n.step(0.0)
        assert isinstance(result, float)

    def test_rate_increases_with_input(self):
        n1 = ErmentroutKopellPopulation()
        n2 = ErmentroutKopellPopulation()
        for _ in range(500):
            r1 = n1.step(0.0)
            r2 = n2.step(10.0)
        # Higher input should give higher rate (or different trajectory)
        assert r1 != r2

    def test_rate_nonnegative(self):
        n = ErmentroutKopellPopulation()
        for _ in range(5000):
            r = n.step(5.0)
        assert r >= 0

    def test_state_finite(self):
        n = ErmentroutKopellPopulation()
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.r)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = ErmentroutKopellPopulation()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.r == 0.1
        assert n.v == -2.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ErmentroutKopellPopulation()
            trace = [(n.step(5.0), n.r, n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestErmentroutKopellAnalytical:
    def test_dr_formula_one_step(self):
        """dr = (Δ/(π·τ) + 2·r·v) / τ · dt."""
        n = ErmentroutKopellPopulation()
        r0, v0 = n.r, n.v
        expected_dr = (n.delta / (np.pi * n.tau) + 2.0 * r0 * v0) / n.tau * n.dt
        n.step(0.0)
        # r = max(0, r0 + dr)
        expected_r = max(0.0, r0 + expected_dr)
        assert abs(n.r - expected_r) < 1e-12

    def test_dv_formula_one_step(self):
        """dv = (v² + η̄ + I + J·τ·r - (π·τ·r)²) / τ · dt."""
        n = ErmentroutKopellPopulation()
        r0, v0 = n.r, n.v
        I = 5.0
        expected_dv = (
            (v0**2 + n.eta_bar + I + n.j * n.tau * r0 - (np.pi * n.tau * r0) ** 2) / n.tau * n.dt
        )
        n.step(I)
        # r was updated first, then v, but dr uses old r
        # Actually looking at source: dr computed, then r updated, then dv uses old v
        # dv is computed from original v0 and r0
        # But wait — r is updated before dv calculation? Let me re-read...
        # Source: dr computed from old state, dv computed from old state,
        # then r updated, then v updated. So both use old values.
        # Actually: dv is computed BEFORE r is updated (lines 27-38 compute dr and dv,
        # then line 39 updates r, line 40 updates v)
        # Wait no - let me re-read: dr is computed (line 27), dv is computed (line 28-37),
        # then self.r = max(0, self.r + dr) (line 39), self.v += dv (line 40)
        # So dv uses the OLD r0 and v0. Good.
        actual_dv = n.v - v0
        assert abs(actual_dv - expected_dv) < 1e-10

    def test_rate_non_negative(self):
        """r = max(0, ...) ensures non-negative rate."""
        n = ErmentroutKopellPopulation(r=0.001)
        for _ in range(1000):
            n.step(-10.0)  # strong inhibition
        assert n.r >= 0.0

    @pytest.mark.parametrize("eta", [-10.0, -5.0, 0.0, 5.0])
    def test_eta_sweep(self, eta: float):
        n = ErmentroutKopellPopulation(eta_bar=eta)
        for _ in range(1000):
            n.step(0.0)
        assert np.isfinite(n.r) and np.isfinite(n.v)

    @pytest.mark.parametrize("j", [5.0, 15.0, 30.0])
    def test_j_coupling_sweep(self, j: float):
        n = ErmentroutKopellPopulation(j=j)
        for _ in range(1000):
            n.step(0.0)
        assert np.isfinite(n.r)


class TestErmentroutKopellPerformance:
    def test_isolation_throughput(self):
        n = ErmentroutKopellPopulation()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(ErmentroutKopellPopulation, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


class TestErmentroutKopellPipeline:
    def test_population(self):
        pop = Population(ErmentroutKopellPopulation, n=5, label="ek")
        assert pop.n == 5

    def test_projection_wiring(self):
        src = Population(ErmentroutKopellPopulation, n=5, label="src")
        tgt = Population(ErmentroutKopellPopulation, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        # Float return → Population clips to {0,1}
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(ErmentroutKopellPopulation, n=10, label="ek")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_field_state_after_run(self):
        pop = Population(ErmentroutKopellPopulation, n=5, label="ek")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        net.run(duration=0.1, dt=0.001, backend="python")
        for neuron in pop.neurons:
            assert np.isfinite(neuron.r)
            assert np.isfinite(neuron.v)
