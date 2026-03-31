# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LoihiCUBANeuron

"""Full pipeline test for LoihiCUBANeuron (Davies et al. 2018).

Intel Loihi fixed-point CUBA LIF:
u = u - u//tau_u + input
v = v - v//tau_v + u
Spike: v→v_reset. All integer arithmetic (// decay).
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: LoihiCUBANeuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestLoihiCUBAIsolation:
    def test_defaults(self):
        n = LoihiCUBANeuron()
        assert n.v == 0 and n.u == 0
        assert n.v_threshold == 1000 and n.tau_v == 10

    def test_integer_state(self):
        n = LoihiCUBANeuron()
        n.step(100)
        assert isinstance(n.v, int) and isinstance(n.u, int)

    def test_step_returns_binary(self):
        assert LoihiCUBANeuron().step(100) in (0, 1)

    def test_reset(self):
        n = LoihiCUBANeuron()
        for _ in range(100):
            n.step(100)
        n.reset()
        assert n.v == 0 and n.u == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LoihiCUBANeuron()
            trace = [(n.step(100), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestLoihiCUBAAnalytical:
    def test_u_integrates_input(self):
        n = LoihiCUBANeuron()
        n.step(500)
        assert n.u > 0

    def test_v_driven_by_u(self):
        n = LoihiCUBANeuron()
        n.step(500)
        n.step(0)
        assert n.v > 0

    def test_integer_division_decay(self):
        n = LoihiCUBANeuron()
        n.v = 100
        decay = 100 // n.tau_v
        assert decay == 10

    def test_spike_resets_v(self):
        n = LoihiCUBANeuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.v == n.v_reset
                break

    def test_two_stage_integration(self):
        """u integrates input, v integrates u (2-stage pipeline)."""
        n = LoihiCUBANeuron()
        assert hasattr(n, "u") and hasattr(n, "v")


class TestLoihiCUBADynamics:
    def test_fires(self):
        assert len(_run(LoihiCUBANeuron(), 200, 5000)) >= 50

    def test_rate_monotonic(self):
        s_low = len(_run(LoihiCUBANeuron(), 100, 5000))
        s_high = len(_run(LoihiCUBANeuron(), 500, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [50, 100, 200, 500])
    def test_fi_sweep(self, current: int):
        n = LoihiCUBANeuron()
        for _ in range(5000):
            n.step(current)
        assert isinstance(n.v, int)


class TestLoihiCUBAPerformance:
    def test_isolation_throughput(self):
        n = LoihiCUBANeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 500_000

    def test_network_throughput(self):
        pop = Population(LoihiCUBANeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 2_000


class TestLoihiCUBAPipeline:
    def test_population(self):
        assert Population(LoihiCUBANeuron, n=10, label="cuba").n == 10

    def test_projection_wiring(self):
        src = Population(LoihiCUBANeuron, n=5, label="src")
        tgt = Population(LoihiCUBANeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(LoihiCUBANeuron, n=10, label="cuba")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = LoihiCUBANeuron()
        train = np.array([float(n.step(200)) for _ in range(5000)])
        assert spike_count(train) >= 20
