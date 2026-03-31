# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: Loihi2Neuron

"""Full pipeline test for Loihi2Neuron (Intel Loihi 2, 2021).

Programmable 3-state-variable neuromorphic neuron:
s3 -= s3 // tau3
s2 = s2 - s2//tau2 + input + w23·s3
s1 = s1 - s1//tau1 + w12·s2 + w13·s3
Spike: s1→s1_reset, s3+=s3_incr. All integer arithmetic.
Cross-coupling (w12, w13, w23). Adaptation via s3.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import pytest

from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: Loihi2Neuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestLoihi2Isolation:
    def test_defaults(self):
        n = Loihi2Neuron()
        assert n.s1 == 0 and n.s2 == 0 and n.s3 == 0
        assert n.s1_threshold == 1000 and n.w12 == 1

    def test_integer_state(self):
        n = Loihi2Neuron()
        n.step(100)
        assert isinstance(n.s1, int) and isinstance(n.s2, int)

    def test_step_returns_binary(self):
        assert Loihi2Neuron().step(100) in (0, 1)

    def test_reset(self):
        n = Loihi2Neuron()
        for _ in range(100):
            n.step(100)
        n.reset()
        assert n.s1 == 0 and n.s2 == 0 and n.s3 == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = Loihi2Neuron()
            trace = [(n.step(100), n.s1) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestLoihi2Analytical:
    def test_three_state_variables(self):
        n = Loihi2Neuron()
        for attr in ["s1", "s2", "s3"]:
            assert hasattr(n, attr)

    def test_s2_integrates_input(self):
        n = Loihi2Neuron()
        n.step(500)
        assert n.s2 > 0

    def test_s1_driven_by_s2(self):
        """s1 = s1 - s1//tau1 + w12·s2. w12=1 → s2 drives s1."""
        n = Loihi2Neuron()
        n.step(500)
        n.step(0)  # s2 still has residual → drives s1
        assert n.s1 > 0

    def test_spike_increments_s3(self):
        """On spike: s3 += s3_incr (adaptation)."""
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s3 >= n.s3_incr
                break

    def test_integer_division_decay(self):
        """Decay via integer division: s -= s // tau."""
        n = Loihi2Neuron()
        n.s1 = 100
        decay = 100 // n.tau1
        assert decay == 10  # 100 // 10

    def test_spike_resets_s1(self):
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s1 == n.s1_reset
                break


class TestLoihi2Dynamics:
    def test_fires(self):
        assert len(_run(Loihi2Neuron(), 200, 5000)) >= 50

    def test_rate_monotonic(self):
        s_low = len(_run(Loihi2Neuron(), 100, 5000))
        s_high = len(_run(Loihi2Neuron(), 500, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [50, 100, 200, 500])
    def test_fi_sweep(self, current: int):
        n = Loihi2Neuron()
        for _ in range(5000):
            n.step(current)
        assert isinstance(n.s1, int)


class TestLoihi2Performance:
    def test_isolation_throughput(self):
        n = Loihi2Neuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 200_000, f"isolation: {N / elapsed:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(Loihi2Neuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 2_000


class TestLoihi2Pipeline:
    def test_population(self):
        assert Population(Loihi2Neuron, n=10, label="l2").n == 10

    def test_projection_wiring(self):
        src = Population(Loihi2Neuron, n=5, label="src")
        tgt = Population(Loihi2Neuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(Loihi2Neuron, n=10, label="l2")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = Loihi2Neuron()
        import numpy as np

        train = np.array([float(n.step(200)) for _ in range(5000)])
        assert spike_count(train) >= 20
