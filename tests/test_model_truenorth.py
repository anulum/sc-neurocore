# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: TrueNorthNeuron

"""Full pipeline test for TrueNorthNeuron (Merolla 2014, IBM TrueNorth).

Digital integer neuron: v += input - leak. Spike at v ≥ threshold.
Negative saturation at v < -threshold → reset. All-integer arithmetic.
Performance benchmarked: ~1.2M isolation steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.truenorth import TrueNorthNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: TrueNorthNeuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestTrueNorthIsolation:
    def test_construction_defaults(self):
        n = TrueNorthNeuron()
        assert n.v == 0
        assert n.leak == 0
        assert n.threshold == 100
        assert n.v_reset == 0

    def test_step_returns_binary(self):
        assert TrueNorthNeuron().step(0) in (0, 1)

    def test_integer_types(self):
        """All state and params are integers (digital neuron)."""
        n = TrueNorthNeuron()
        assert isinstance(n.v, int)
        assert isinstance(n.threshold, int)
        assert isinstance(n.leak, int)

    def test_state_evolves(self):
        n = TrueNorthNeuron()
        n.step(50)
        assert n.v == 50  # 0 + 50 - 0 = 50

    def test_reset(self):
        n = TrueNorthNeuron()
        for _ in range(20):
            n.step(50)
        n.reset()
        assert n.v == 0


class TestTrueNorthIntegerArithmetic:
    """Core: v += input - leak. All integer. No floating point."""

    def test_voltage_accumulation_exact(self):
        """v = sum(input) - steps*leak when no spikes occur."""
        n = TrueNorthNeuron(leak=0, threshold=1000)
        for _ in range(10):
            n.step(7)
        assert n.v == 70  # 10 * 7 = 70

    def test_leak_subtracted_each_step(self):
        """v += input - leak. With leak=5, input=10: net +5 per step."""
        n = TrueNorthNeuron(leak=5, threshold=1000)
        for _ in range(10):
            n.step(10)
        assert n.v == 50  # 10 * (10 - 5) = 50

    def test_leak_exceeds_input_no_spikes(self):
        """When leak > input, v decreases → never reaches threshold."""
        n = TrueNorthNeuron(leak=50)
        spikes = sum(n.step(20) for _ in range(1000))
        assert spikes == 0

    def test_spike_rate_exact(self):
        """With leak=0, input=I: spike every ceil(threshold/I) steps.

        I=10, θ=100: spike every 10 steps → 100 spikes/1000.
        """
        n = TrueNorthNeuron(leak=0, threshold=100)
        outputs = [n.step(10) for _ in range(1000)]
        assert outputs.count(1) == 100

    def test_spike_resets_to_v_reset(self):
        n = TrueNorthNeuron(threshold=100, v_reset=0)
        for _ in range(1000):
            s = n.step(50)
            if s == 1:
                assert n.v == 0
                break
        else:
            pytest.fail("No spike")

    def test_negative_saturation(self):
        """v < -threshold → reset to v_reset (prevents unbounded negative)."""
        n = TrueNorthNeuron(threshold=100)
        n.step(-200)  # v = -200 < -100
        assert n.v == 0  # reset

    def test_negative_saturation_boundary(self):
        """v = -100: NOT reset (condition is v < -threshold, not ≤)."""
        n = TrueNorthNeuron(threshold=100)
        n.v = -100
        n.step(0)  # v stays -100, check: -100 < -100 is False
        assert n.v == -100

    def test_custom_v_reset(self):
        n = TrueNorthNeuron(threshold=100, v_reset=10)
        for _ in range(1000):
            if n.step(50) == 1:
                assert n.v == 10
                break


class TestTrueNorthLeakEffect:
    def test_leak_reduces_effective_rate(self):
        """Higher leak → lower effective current → fewer spikes."""
        n_noleak = TrueNorthNeuron(leak=0)
        n_leak = TrueNorthNeuron(leak=10)
        s_noleak = sum(n_noleak.step(20) for _ in range(1000))
        s_leak = sum(n_leak.step(20) for _ in range(1000))
        assert s_noleak > s_leak

    def test_analytical_rate_with_leak(self):
        """Rate = steps / ceil(θ / (I - leak)) when I > leak."""
        n = TrueNorthNeuron(leak=10, threshold=100)
        I = 20
        steps = 1000
        spikes = sum(n.step(I) for _ in range(steps))
        effective = I - n.leak  # 10
        expected = steps * effective // n.threshold
        assert abs(spikes - expected) <= 2


class TestTrueNorthFI:
    def test_zero_input_silent(self):
        n = TrueNorthNeuron()
        assert sum(n.step(0) for _ in range(1000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [10, 20, 50, 100]:
            n = TrueNorthNeuron()
            rates.append(sum(n.step(I) for _ in range(1000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_suprathreshold_every_step(self):
        """I ≥ threshold → spike every step."""
        n = TrueNorthNeuron()
        spikes = sum(n.step(100) for _ in range(100))
        assert spikes == 100


class TestTrueNorthPerformance:
    def test_isolation_throughput(self):
        n = TrueNorthNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50)
        elapsed = time.perf_counter() - t0
        steps_per_s = N / elapsed
        assert steps_per_s > 100000, f"{steps_per_s:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(TrueNorthNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 50 * 500
        nsteps_per_s = neuron_steps / elapsed
        assert nsteps_per_s > 5000, f"{nsteps_per_s:.0f} neuron-steps/s"


class TestTrueNorthPipeline:
    def test_population(self):
        assert Population(TrueNorthNeuron, n=10, label="tn").n == 10

    def test_network_with_drive(self):
        pop = Population(TrueNorthNeuron, n=10, label="tn")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(TrueNorthNeuron, n=10, label="src")
        tgt = Population(TrueNorthNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self):
        n = TrueNorthNeuron()
        train = np.array([float(n.step(50)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 10000 * 0.001
        assert abs(rate - sc / duration) < 10.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TrueNorthNeuron()
            trace = [(n.step(50), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
