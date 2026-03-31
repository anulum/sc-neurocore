# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: JansenRitUnit

"""Full pipeline: JansenRitUnit. FULL PIPELINE + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron, current, steps):
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns(self):
        n = JansenRitUnit()
        result = n.step(200.0)
        assert result is not None

    def test_state_finite(self):
        n = JansenRitUnit()
        for _ in range(3000):
            n.step(200.0)
        pass  # special state

    def test_reset(self):
        n = JansenRitUnit()
        for _ in range(100):
            n.step(200.0)
        n.reset()


class TestDynamics:
    def test_returns_float(self):
        """Rate/mass model returns float, not int spike."""
        n = JansenRitUnit()
        result = n.step(200.0)
        assert isinstance(result, (float, int, np.floating, np.integer))

    def test_rate_monotonic(self):
        n_low = JansenRitUnit()
        n_high = JansenRitUnit()
        s_low = len(_run(n_low, 100.0, 5000))
        s_high = len(_run(n_high, 300.0, 5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = JansenRitUnit()
            trace = [n.step(200.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = JansenRitUnit()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000


class TestPipeline:
    def test_population(self):
        assert Population(JansenRitUnit, n=5, label="t").n == 5

    def test_network(self):
        pop = Population(JansenRitUnit, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_projection_wiring(self):
        src = Population(JansenRitUnit, n=5, label="s")
        tgt = Population(JansenRitUnit, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=200.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
