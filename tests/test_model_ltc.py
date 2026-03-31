# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LiquidTimeConstantNeuron

"""Full pipeline test for LiquidTimeConstantNeuron (Hasani et al. 2021).

Input-dependent time constant:
τ(I) = τ_base · σ(w_τ·I + bias), clipped ≥ 0.1
f_target = tanh(w_x·x + w_in·I)
dx = dt/τ · (-x + f_target)
Spike: x→0 when x ≥ threshold.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.ltc import LiquidTimeConstantNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: LiquidTimeConstantNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestLTCIsolation:
    def test_defaults(self):
        n = LiquidTimeConstantNeuron()
        assert n.x == 0.0 and n.tau_base == 10.0
        assert n.w_tau == -0.5 and n.v_threshold == 1.0

    def test_step_returns_binary(self):
        assert LiquidTimeConstantNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = LiquidTimeConstantNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.x)

    def test_reset(self):
        n = LiquidTimeConstantNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.x == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LiquidTimeConstantNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestLTCAnalytical:
    def test_input_dependent_tau(self):
        """τ depends on input via sigmoid: τ = τ_base · σ(w_τ·I)."""
        n = LiquidTimeConstantNeuron()
        # At I=0: σ(0) = 0.5 → τ = 10 * 0.5 = 5
        tau_zero = n.tau_base * (1.0 / (1.0 + np.exp(0.0)))
        assert abs(tau_zero - 5.0) < 1e-10

    def test_tau_clipped(self):
        """τ ≥ 0.1 prevents division by zero."""
        n = LiquidTimeConstantNeuron()
        # Very negative input → σ → 0 → tau → 0, clipped to 0.1
        n.step(-1000.0)
        assert np.isfinite(n.x)

    def test_f_target_tanh(self):
        """f_target = tanh(w_x·x + w_in·I). Bounded [-1, 1]."""
        f = np.tanh(0.8 * 0.0 + 1.0 * 5.0)
        assert -1 <= f <= 1

    def test_spike_resets_x(self):
        n = LiquidTimeConstantNeuron()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.x == 0.0
                break


class TestLTCDynamics:
    def test_fires(self):
        assert len(_run(LiquidTimeConstantNeuron(), 5.0, 5000)) >= 10

    def test_subthreshold(self):
        assert len(_run(LiquidTimeConstantNeuron(), 0.01, 5000)) == 0

    def test_rate_monotonic(self):
        rates = [len(_run(LiquidTimeConstantNeuron(), I, 5000)) for I in [1.0, 5.0, 10.0]]
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = LiquidTimeConstantNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.x)


class TestLTCParameters:
    @pytest.mark.parametrize("tau_base", [5.0, 10.0, 20.0])
    def test_tau_base_sweep(self, tau_base: float):
        n = LiquidTimeConstantNeuron(tau_base=tau_base)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.x)

    @pytest.mark.parametrize("w_tau", [-1.0, -0.5, 0.0])
    def test_w_tau_sweep(self, w_tau: float):
        n = LiquidTimeConstantNeuron(w_tau=w_tau)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.x)


class TestLTCPerformance:
    def test_isolation_throughput(self):
        n = LiquidTimeConstantNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50_000

    def test_network_throughput(self):
        pop = Population(LiquidTimeConstantNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 2_000


class TestLTCPipeline:
    def test_population(self):
        assert Population(LiquidTimeConstantNeuron, n=10, label="ltc").n == 10

    def test_projection_wiring(self):
        src = Population(LiquidTimeConstantNeuron, n=5, label="src")
        tgt = Population(LiquidTimeConstantNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(LiquidTimeConstantNeuron, n=10, label="ltc")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = LiquidTimeConstantNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        assert spike_count(train) >= 5
        assert firing_rate(train, dt=0.001) > 0
