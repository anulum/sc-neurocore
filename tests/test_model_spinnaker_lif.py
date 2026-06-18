# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SpiNNakerLIFNeuron

"""Full pipeline test for SpiNNakerLIFNeuron (Furber 2014).

ARM Cortex-M4 digital LIF with absolute refractory period.
Performance: ~1.8M isolation steps/s."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.spinnaker_lif import SpiNNakerLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: SpiNNakerLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSpiNNakerLIFIsolation:
    def test_defaults(self):
        n = SpiNNakerLIFNeuron()
        assert n.v == -70.0 and n.tau_m == 20.0 and n.tau_refrac == 2.0

    def test_step_returns_binary(self):
        assert SpiNNakerLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = SpiNNakerLIFNeuron()
        for _ in range(50000):
            n.step(25.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = SpiNNakerLIFNeuron()
        for _ in range(100):
            n.step(25.0)
        n.reset()
        assert n.v == n.v_rest and n.refrac_count == 0.0

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="tau_m must be positive"):
            SpiNNakerLIFNeuron(tau_m=0.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            SpiNNakerLIFNeuron(dt=0.0)
        with pytest.raises(ValueError, match="refrac_count must be non-negative"):
            SpiNNakerLIFNeuron(refrac_count=-1.0)

    def test_rejects_invalid_current_without_mutation(self):
        n = SpiNNakerLIFNeuron()
        v0 = n.v
        with pytest.raises(ValueError, match="current must be finite"):
            n.step(float("nan"))
        assert n.v == v0


class TestSpiNNakerLIFRefractory:
    def test_refractory_blocks_spikes(self):
        """During refractory period (tau_refrac=2), no spikes can occur."""
        n = SpiNNakerLIFNeuron()
        # Drive to spike
        for _ in range(100):
            if n.step(50.0) == 1:
                # Immediately after spike: refrac_count = 2
                assert n.refrac_count == n.tau_refrac
                # Next 2 steps should be blocked
                s1 = n.step(50.0)
                s2 = n.step(50.0)
                assert s1 == 0 and s2 == 0, "Should be refractory"
                return
        raise AssertionError("No spike")

    def test_refractory_reduces_rate(self):
        """Refractory period limits maximum firing rate."""
        n_norefrac = SpiNNakerLIFNeuron(tau_refrac=0.0)
        n_refrac = SpiNNakerLIFNeuron(tau_refrac=5.0)
        s_no = len(_run(n_norefrac, current=50.0, steps=5000))
        s_yes = len(_run(n_refrac, current=50.0, steps=5000))
        assert s_no > s_yes

    def test_refrac_count_decrements(self):
        n = SpiNNakerLIFNeuron(tau_refrac=3.0)
        n.refrac_count = 3.0
        n.step(0.0)
        assert n.refrac_count == 2.0
        n.step(0.0)
        assert n.refrac_count == 1.0

    def test_i_offset_adds_baseline(self):
        """i_offset provides constant baseline current."""
        n = SpiNNakerLIFNeuron(i_offset=25.0)
        spikes = len(_run(n, current=0.0, steps=5000))
        assert spikes > 0, "i_offset should drive spikes even at I=0"


class TestSpiNNakerLIFDynamics:
    def test_membrane_equation(self):
        """Exact LIF flow solves constant-current membrane dynamics."""
        n = SpiNNakerLIFNeuron(tau_refrac=0.0)
        v0 = n.v
        current = 15.0
        n.step(current)
        steady = n.v_rest + current + n.i_offset
        expected = steady + (v0 - steady) * math.exp(-n.dt / n.tau_m)
        assert abs(n.v - expected) < 1e-10

    def test_exact_flow_reduces_to_euler_order_for_small_dt(self):
        n = SpiNNakerLIFNeuron(dt=1.0e-6, tau_refrac=0.0)
        v0 = n.v
        current = 15.0
        n.step(current)
        euler = v0 + (-(v0 - n.v_rest) + current + n.i_offset) / n.tau_m * n.dt
        assert abs(n.v - euler) < 1e-12

    def test_steady_state(self):
        """V_ss = V_rest + I. At I=10: V_ss = -60, below threshold."""
        n = SpiNNakerLIFNeuron()
        for _ in range(10000):
            n.step(10.0)
        assert abs(n.v - (-60.0)) < 0.1

    def test_monotonic_fi(self):
        rates = []
        for I in [25.0, 30.0, 40.0, 50.0]:
            n = SpiNNakerLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestSpiNNakerLIFPerformance:
    def test_isolation_throughput(self):
        n = SpiNNakerLIFNeuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(25.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 100000

    def test_network_throughput(self):
        pop = Population(SpiNNakerLIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=25.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestSpiNNakerLIFPipeline:
    def test_population(self):
        assert Population(SpiNNakerLIFNeuron, n=10, label="snlif").n == 10

    def test_network_spikes(self):
        pop = Population(SpiNNakerLIFNeuron, n=10, label="snlif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(SpiNNakerLIFNeuron, n=10, label="src")
        tgt = Population(SpiNNakerLIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=25.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = SpiNNakerLIFNeuron()
        train = np.array([float(n.step(25.0)) for _ in range(5000)])
        assert spike_count(train) >= 10
        assert firing_rate(train, dt=0.001) > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SpiNNakerLIFNeuron()
            trace = [(n.step(25.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
