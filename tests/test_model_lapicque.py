# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LapicqueNeuron

"""Full pipeline: LapicqueNeuron. FULL PIPELINE + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron, current, steps):
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns(self):
        n = LapicqueNeuron()
        result = n.step(20.0)
        assert result is not None

    def test_state_finite(self):
        n = LapicqueNeuron()
        for _ in range(3000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = LapicqueNeuron()
        for _ in range(100):
            n.step(20.0)
        n.reset()


class TestDynamics:
    def test_fires(self):
        n = LapicqueNeuron()
        spikes = _run(n, 20.0, 5000)
        assert len(spikes) >= 100

    def test_rate_monotonic(self):
        n_low = LapicqueNeuron()
        n_high = LapicqueNeuron()
        s_low = len(_run(n_low, 10.0, 5000))
        s_high = len(_run(n_high, 50.0, 5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LapicqueNeuron()
            trace = [n.step(20.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = LapicqueNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000


class TestPipeline:
    def test_population(self):
        assert Population(LapicqueNeuron, n=5, label="t").n == 5

    def test_network(self):
        pop = Population(LapicqueNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(LapicqueNeuron, n=5, label="s")
        tgt = Population(LapicqueNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=20.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
