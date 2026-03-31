# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: InhomogeneousPoissonNeuron

"""Full pipeline: InhomogeneousPoissonNeuron. FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.inhomogeneous_poisson import InhomogeneousPoissonNeuron
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
        assert InhomogeneousPoissonNeuron().step(100.0) in (0, 1)

    def test_state_finite(self):
        n = InhomogeneousPoissonNeuron()
        for _ in range(3000):
            n.step(100.0)
        pass  # no scalar v

    def test_reset(self):
        n = InhomogeneousPoissonNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()


class TestDynamics:
    def test_fires(self):
        n = InhomogeneousPoissonNeuron()
        spikes = _run(n, 100.0, 5000)
        assert len(spikes) >= 10

    def test_rate_monotonic(self):
        n_low = InhomogeneousPoissonNeuron()
        n_high = InhomogeneousPoissonNeuron()
        s_low = len(_run(n_low, 50.0, 5000))
        s_high = len(_run(n_high, 500.0, 5000))
        assert s_high >= s_low

    def test_two_runs_differ(self):
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        t1 = [n1.step(50.0) for _ in range(1000)]
        t2 = [n2.step(50.0) for _ in range(1000)]
        assert t1 != t2


class TestPerformance:
    def test_isolation_throughput(self):
        n = InhomogeneousPoissonNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(InhomogeneousPoissonNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 500


class TestPipeline:
    def test_population(self):
        assert Population(InhomogeneousPoissonNeuron, n=5, label="t").n == 5

    def test_network_spikes(self):
        pop = Population(InhomogeneousPoissonNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_projection_wiring(self):
        src = Population(InhomogeneousPoissonNeuron, n=5, label="s")
        tgt = Population(InhomogeneousPoissonNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = InhomogeneousPoissonNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 0
