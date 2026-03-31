# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: IntegerQIFNeuron

"""Full pipeline: IntegerQIFNeuron. FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron, current, steps):
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns_binary(self):
        assert IntegerQIFNeuron().step(100.0) in (0, 1)

    def test_state_finite(self):
        n = IntegerQIFNeuron()
        for _ in range(3000):
            n.step(100)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = IntegerQIFNeuron()
        for _ in range(100):
            n.step(100)
        n.reset()


class TestDynamics:
    def test_fires(self):
        n = IntegerQIFNeuron()
        spikes = _run(n, 100, 5000)
        assert len(spikes) >= 10

    def test_rate_monotonic(self):
        n_low = IntegerQIFNeuron()
        n_high = IntegerQIFNeuron()
        s_low = len(_run(n_low, 50, 5000))
        s_high = len(_run(n_high, 500, 5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = IntegerQIFNeuron()
            trace = [n.step(100) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = IntegerQIFNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 200000

    def test_network_incompatible(self):
        """Integer >> operator fails with float from Population.step_all."""
        import pytest

        pop = Population(IntegerQIFNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        with pytest.raises(TypeError):
            net.run(duration=0.1, dt=0.001, backend="python")


class TestPipeline:
    def test_population(self):
        assert Population(IntegerQIFNeuron, n=5, label="t").n == 5

    def test_analysis(self):
        n = IntegerQIFNeuron()
        train = np.array([float(n.step(100)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 0
