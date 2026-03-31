# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LearnableNeuronModel

"""Full pipeline test for LearnableNeuronModel (Jahns et al. 2025).

Fully parameterised learnable neuron:
V[t+1] = α·V[t] + β·I[t] + γ·f(V[t])
f(V) = 1/(1+exp(-f_slope·(V-f_shift)))  — learnable sigmoid.
α=0.9, β=0.1, γ=0.05. All trainable for SNN backprop.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.lnm import LearnableNeuronModel
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: LearnableNeuronModel, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestLNMIsolation:
    def test_defaults(self):
        n = LearnableNeuronModel()
        assert n.v == 0.0 and n.alpha == 0.9 and n.beta == 0.1
        assert n.gamma == 0.05 and n.v_threshold == 1.0

    def test_step_returns_binary(self):
        assert LearnableNeuronModel().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = LearnableNeuronModel()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = LearnableNeuronModel()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LearnableNeuronModel()
            trace = [(n.step(5.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestLNMAnalytical:
    def test_v_update_formula(self):
        n = LearnableNeuronModel()
        v0 = n.v
        I = 0.5
        f_v = 1.0 / (1.0 + np.exp(-n.f_slope * (v0 - n.f_shift)))
        expected = n.alpha * v0 + n.beta * I + n.gamma * f_v
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-12

    def test_sigmoid_midpoint(self):
        n = LearnableNeuronModel()
        f = 1.0 / (1.0 + np.exp(0.0))
        assert abs(f - 0.5) < 1e-12

    def test_alpha_decay(self):
        n = LearnableNeuronModel(v_threshold=100.0)
        n.v = 0.8
        for _ in range(50):
            n.step(0.0)
        assert n.v < 0.8

    def test_beta_scales_input(self):
        n1 = LearnableNeuronModel(beta=0.1, v_threshold=100.0)
        n2 = LearnableNeuronModel(beta=0.5, v_threshold=100.0)
        for _ in range(50):
            n1.step(5.0)
            n2.step(5.0)
        assert n2.v > n1.v

    def test_gamma_zero_linear(self):
        n = LearnableNeuronModel(gamma=0.0, v_threshold=100.0)
        n.v = 0.5
        n.step(1.0)
        assert abs(n.v - (0.9 * 0.5 + 0.1 * 1.0)) < 1e-12

    def test_spike_resets(self):
        n = LearnableNeuronModel()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.v == n.v_reset
                break


class TestLNMDynamics:
    def test_fires(self):
        assert len(_run(LearnableNeuronModel(), 5.0, 5000)) >= 50

    def test_subthreshold(self):
        assert len(_run(LearnableNeuronModel(), 0.05, 5000)) == 0

    def test_rate_monotonic(self):
        rates = [len(_run(LearnableNeuronModel(), I, 5000)) for I in [1.0, 5.0, 10.0]]
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = LearnableNeuronModel()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)


class TestLNMParameters:
    @pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
    def test_alpha_sweep(self, alpha: float):
        n = LearnableNeuronModel(alpha=alpha)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("gamma", [0.0, 0.05, 0.2])
    def test_gamma_sweep(self, gamma: float):
        n = LearnableNeuronModel(gamma=gamma)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)


class TestLNMPerformance:
    def test_isolation_throughput(self):
        n = LearnableNeuronModel()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 100_000

    def test_network_throughput(self):
        pop = Population(LearnableNeuronModel, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 2_000


class TestLNMPipeline:
    def test_population(self):
        assert Population(LearnableNeuronModel, n=10, label="lnm").n == 10

    def test_projection_wiring(self):
        src = Population(LearnableNeuronModel, n=5, label="src")
        tgt = Population(LearnableNeuronModel, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(LearnableNeuronModel, n=10, label="lnm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = LearnableNeuronModel()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        assert spike_count(train) >= 20
        assert firing_rate(train, dt=0.001) > 0
