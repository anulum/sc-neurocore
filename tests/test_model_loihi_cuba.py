# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LoihiCUBANeuron

"""Full pipeline: LoihiCUBANeuron. FULL PIPELINE + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron, current, steps):
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns(self):
        n = LoihiCUBANeuron()
        result = n.step(1000)
        assert result is not None

    def test_state_finite(self):
        n = LoihiCUBANeuron()
        for _ in range(3000):
            n.step(1000)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = LoihiCUBANeuron()
        for _ in range(100):
            n.step(1000)
        n.reset()


class TestDynamics:
    def test_fires(self):
        n = LoihiCUBANeuron()
        spikes = _run(n, 1000, 5000)
        assert len(spikes) >= 100

    def test_rate_monotonic(self):
        n_low = LoihiCUBANeuron()
        n_high = LoihiCUBANeuron()
        s_low = len(_run(n_low, 500, 5000))
        s_high = len(_run(n_high, 2000, 5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LoihiCUBANeuron()
            trace = [n.step(1000) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = LoihiCUBANeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1000)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 200000


class TestPipeline:
    def test_population(self):
        assert Population(LoihiCUBANeuron, n=5, label="t").n == 5

    def test_network(self):
        pop = Population(LoihiCUBANeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=float(1000), dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(LoihiCUBANeuron, n=5, label="s")
        tgt = Population(LoihiCUBANeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=float(1000), dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=float(1000), probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
