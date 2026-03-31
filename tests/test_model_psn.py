# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ParallelSpikingNeuron (PSN)

"""Full pipeline test for ParallelSpikingNeuron (PSN, 2024).

1D convolution kernel over circular buffer. Spike when
dot(kernel, buffer) >= threshold. Buffer cleared on spike."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.psn import ParallelSpikingNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


class TestPSNIsolation:
    def test_construction_defaults(self):
        n = ParallelSpikingNeuron()
        assert n.kernel_size == 8
        assert n.v_threshold == 1.0
        assert n.kernel.shape == (8,)
        assert n.buffer.shape == (8,)

    def test_step_returns_binary(self):
        assert ParallelSpikingNeuron().step(0.0) in (0, 1)

    def test_default_kernel_is_uniform(self):
        """Default kernel = 1/kernel_size (averaging filter)."""
        n = ParallelSpikingNeuron(kernel_size=4)
        np.testing.assert_allclose(n.kernel, [0.25, 0.25, 0.25, 0.25])

    def test_buffer_fills_circularly(self):
        """Input values are written to circular buffer."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=100.0)
        for i in range(6):
            n.step(float(i))
        # Buffer wraps: positions 0,1,2,3 → values 4,5,2,3
        assert n.buffer[0] == 4.0
        assert n.buffer[1] == 5.0

    def test_reset(self):
        n = ParallelSpikingNeuron()
        for _ in range(20):
            n.step(5.0)
        n.reset()
        assert np.all(n.buffer == 0.0)
        assert n._ptr == 0


class TestPSNScoring:
    def test_spike_at_threshold(self):
        """With uniform kernel: score = mean(buffer). Spike when mean >= θ."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        # Fill buffer with 1.0 → score = mean([1,1,1,1]) = 1.0 → spike
        for i in range(3):
            n.step(1.0)
        s = n.step(1.0)  # 4th step fills buffer → score = 1.0 → spike
        assert s == 1

    def test_subthreshold_no_spike(self):
        """Score below threshold → no spike."""
        n = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        # I=0.5 → avg=0.5 < 1.0
        spikes = sum(n.step(0.5) for _ in range(100))
        assert spikes == 0

    def test_buffer_cleared_on_spike(self):
        """After spike, buffer is zeroed → next step starts fresh."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        # Fill and trigger spike
        for _ in range(4):
            n.step(2.0)
        # Buffer should now be zeros
        assert np.all(n.buffer == 0.0)

    def test_rate_proportional_to_input(self):
        """At I=threshold: spikes every kernel_size steps (refill cycle)."""
        n = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        spikes = sum(n.step(1.0) for _ in range(500))
        # Spikes every 8 steps (fill buffer, spike, clear, repeat)
        expected = 500 // 8
        assert abs(spikes - expected) <= 2

    def test_double_input_double_rate(self):
        """I=2*θ → score reaches threshold with half the buffer filled."""
        n1 = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        n2 = ParallelSpikingNeuron(kernel_size=8, v_threshold=1.0)
        s1 = sum(n1.step(1.0) for _ in range(500))
        s2 = sum(n2.step(2.0) for _ in range(500))
        assert s2 > s1


class TestPSNCustomKernel:
    def test_custom_kernel_affects_scoring(self):
        """Non-uniform kernel weights recent inputs differently."""
        n = ParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        n.kernel = np.array([0.0, 0.0, 0.0, 1.0])  # only last entry
        # Only the value at position 3 matters
        n.step(0.0)
        n.step(0.0)
        n.step(0.0)
        s = n.step(1.0)  # pos 3 gets 1.0, score = 1.0
        assert s == 1


class TestPSNEdgeCases:
    @pytest.mark.parametrize("ks", [2, 4, 8, 16])
    def test_kernel_size_variations(self, ks: int):
        n = ParallelSpikingNeuron(kernel_size=ks, v_threshold=1.0)
        spikes = sum(n.step(1.0) for _ in range(500))
        assert spikes > 0

    def test_zero_input(self):
        n = ParallelSpikingNeuron()
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ParallelSpikingNeuron()
            trace = [(n.step(1.5), float(n.buffer.sum())) for _ in range(50)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPSNPerformance:
    def test_isolation_throughput(self):
        import time

        n = ParallelSpikingNeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 100_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        import time

        pop = Population(ParallelSpikingNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


class TestPSNPipeline:
    def test_population(self):
        assert Population(ParallelSpikingNeuron, n=10, label="psn").n == 10

    def test_projection_wiring(self):
        src = Population(ParallelSpikingNeuron, n=5, label="src")
        tgt = Population(ParallelSpikingNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(ParallelSpikingNeuron, n=10, label="psn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = ParallelSpikingNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(500)])
        assert spike_count(train) > 10

    def test_analysis_spike_count_consistency(self):
        n = ParallelSpikingNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(500)])
        assert spike_count(train) == int(train.sum())

    def test_analysis_isi(self):
        n = ParallelSpikingNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(500)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = ParallelSpikingNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(500)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_analysis_cross_validation(self):
        n = ParallelSpikingNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(500)])
        sc = spike_count(train)
        dt_sim = 0.001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
