# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DurstewitzDopamineNeuron

"""Full pipeline test for DurstewitzDopamineNeuron.

Fires spontaneously at I=0 (~20 spikes/10k). Rate increases with I.
Performance: ~54K steps/s. Full pipeline wired."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: DurstewitzDopamineNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestDurstewitzIsolation:
    def test_step_returns_binary(self):
        assert DurstewitzDopamineNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = DurstewitzDopamineNeuron()
        for _ in range(50000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = DurstewitzDopamineNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert np.isfinite(n.v)


class TestDurstewitzDynamics:
    def test_spontaneous_firing(self):
        """Fires at I=0 (dopaminergic tonic activity)."""
        n = DurstewitzDopamineNeuron()
        spikes = _run(n, current=0.0, steps=10000)
        assert len(spikes) >= 5

    def test_rate_increases_with_current(self):
        n0 = DurstewitzDopamineNeuron()
        n50 = DurstewitzDopamineNeuron()
        s0 = len(_run(n0, current=0.0, steps=10000))
        s50 = len(_run(n50, current=50.0, steps=10000))
        assert s50 > s0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.0, 10.0, 30.0, 50.0]:
            n = DurstewitzDopamineNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DurstewitzDopamineNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestDurstewitzPerformance:
    def test_isolation_throughput(self):
        n = DurstewitzDopamineNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000


class TestDurstewitzPipeline:
    def test_population(self):
        assert Population(DurstewitzDopamineNeuron, n=5, label="dd").n == 5

    def test_network_spikes(self):
        pop = Population(DurstewitzDopamineNeuron, n=5, label="dd")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(DurstewitzDopamineNeuron, n=5, label="src")
        tgt = Population(DurstewitzDopamineNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = DurstewitzDopamineNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
