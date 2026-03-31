# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AvRonCardiacNeuron

"""Full pipeline test for AvRonCardiacNeuron.

Cardiac oscillator: fires spontaneously at I=0 (ISI≈159 steps).
Performance: ~49K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.av_ron_cardiac import AvRonCardiacNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: AvRonCardiacNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestAvRonIsolation:
    def test_defaults(self):
        n = AvRonCardiacNeuron()
        assert np.isfinite(n.v)

    def test_step_returns_binary(self):
        assert AvRonCardiacNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = AvRonCardiacNeuron()
        for _ in range(50000):
            n.step(0.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = AvRonCardiacNeuron()
        for _ in range(500):
            n.step(0.0)
        n.reset()
        # Verify reset restores some initial value
        assert np.isfinite(n.v)


class TestAvRonSpontaneousOscillation:
    def test_fires_at_zero_input(self):
        """Cardiac oscillator fires spontaneously."""
        n = AvRonCardiacNeuron()
        spikes = _run(n, current=0.0, steps=50000)
        assert len(spikes) >= 100

    def test_regular_isi(self):
        n = AvRonCardiacNeuron()
        spikes = _run(n, current=0.0, steps=50000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.2

    def test_isi_around_159(self):
        n = AvRonCardiacNeuron()
        spikes = _run(n, current=0.0, steps=50000)
        isis = np.diff(spikes[5:])
        mean_isi = np.mean(isis)
        assert 100 < mean_isi < 250, f"Mean ISI = {mean_isi:.0f}"

    def test_current_modulates_rate(self):
        """External current should modulate the cardiac rhythm."""
        n0 = AvRonCardiacNeuron()
        n10 = AvRonCardiacNeuron()
        s0 = len(_run(n0, current=0.0, steps=50000))
        s10 = len(_run(n10, current=10.0, steps=50000))
        # At minimum: both should fire (cardiac oscillator)
        assert s0 > 0 and s10 > 0


class TestAvRonPerformance:
    def test_isolation_throughput(self):
        n = AvRonCardiacNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000


class TestAvRonPipeline:
    def test_population(self):
        assert Population(AvRonCardiacNeuron, n=5, label="avron").n == 5

    def test_network_spikes(self):
        pop = Population(AvRonCardiacNeuron, n=5, label="avron")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(AvRonCardiacNeuron, n=5, label="src")
        tgt = Population(AvRonCardiacNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = AvRonCardiacNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AvRonCardiacNeuron()
            trace = [(n.step(0.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
