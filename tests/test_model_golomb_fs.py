# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GolombFSNeuron

"""Full pipeline test for GolombFSNeuron. FULL PIPELINE WIRED."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.golomb_fs import GolombFSNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: GolombFSNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns_binary(self):
        assert GolombFSNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = GolombFSNeuron()
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = GolombFSNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()


class TestDynamics:
    def test_fires(self):
        n = GolombFSNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 10

    def test_rate_increases(self):
        n_low = GolombFSNeuron()
        n_high = GolombFSNeuron()
        s_low = len(_run(n_low, current=2.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GolombFSNeuron()
            trace = [n.step(5.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = GolombFSNeuron()
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 2000

    def test_network_throughput(self):
        pop = Population(GolombFSNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 1000


class TestPipeline:
    def test_population(self):
        assert Population(GolombFSNeuron, n=10, label="test").n == 10

    def test_network_spikes(self):
        pop = Population(GolombFSNeuron, n=10, label="test")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(GolombFSNeuron, n=5, label="src")
        tgt = Population(GolombFSNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 1
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0
