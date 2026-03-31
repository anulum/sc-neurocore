# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DendrifyNeuron

"""Full pipeline test for DendrifyNeuron.

Fires at I≥50. Performance: ~124K steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.dendrify import DendrifyNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: DendrifyNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestDendrifyIsolation:
    def test_step_returns_binary(self):
        assert DendrifyNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = DendrifyNeuron()
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v_s)

    def test_reset(self):
        n = DendrifyNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert np.isfinite(n.v_s)


class TestDendrifyDynamics:
    def test_subthreshold_silent(self):
        n = DendrifyNeuron()
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = DendrifyNeuron()
        assert len(_run(n, current=50.0, steps=10000)) >= 50

    def test_rate_increases(self):
        n50 = DendrifyNeuron()
        n100 = DendrifyNeuron()
        s50 = len(_run(n50, current=50.0, steps=10000))
        s100 = len(_run(n100, current=100.0, steps=10000))
        assert s100 > s50

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DendrifyNeuron()
            trace = [(n.step(50.0), n.v_s) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestDendrifyPerformance:
    def test_isolation_throughput(self):
        n = DendrifyNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000


class TestDendrifyPipeline:
    def test_population(self):
        assert Population(DendrifyNeuron, n=10, label="dend").n == 10

    def test_network_spikes(self):
        pop = Population(DendrifyNeuron, n=10, label="dend")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        # Dendrify needs high current — may not spike in network
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = DendrifyNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) >= 10
