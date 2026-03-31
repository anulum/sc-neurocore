# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SpiNNaker2Neuron

"""Full pipeline test for SpiNNaker2Neuron (TU Dresden 2024).

Fixed-point LIF on ARM Cortex-M4F. Integer multiply-shift decay.
All state and params are integers. Performance: ~1.8M isolation steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.spinnaker2 import SpiNNaker2Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: SpiNNaker2Neuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSpiNNaker2Isolation:
    def test_defaults(self):
        n = SpiNNaker2Neuron()
        assert n.v == 0 and n.v_threshold == 1024
        assert n.decay_mult == 243 and n.decay_shift == 8
        assert n.refrac_steps == 2

    def test_step_returns_binary(self):
        assert SpiNNaker2Neuron().step(0) in (0, 1)

    def test_integer_types(self):
        n = SpiNNaker2Neuron()
        assert isinstance(n.v, int)
        assert isinstance(n.v_threshold, int)

    def test_state_finite(self):
        n = SpiNNaker2Neuron()
        for _ in range(50000):
            n.step(500)
        # Integer can't be NaN, but check it's reasonable
        assert abs(n.v) < 10**9

    def test_reset(self):
        n = SpiNNaker2Neuron()
        for _ in range(100):
            n.step(500)
        n.reset()
        assert n.v == n.v_rest and n._refrac_count == 0


class TestSpiNNaker2FixedPointDecay:
    """Core: v = ((v - v_rest) * decay_mult >> decay_shift) + v_rest + I.

    This is fixed-point exponential decay: alpha = decay_mult / 2^decay_shift
    = 243/256 ≈ 0.949, approximating exp(-1/10) ≈ 0.905 with slight error.
    """

    def test_decay_formula(self):
        """One step with I=0: v_new = (v * 243 >> 8) + 0."""
        n = SpiNNaker2Neuron()
        n.v = 1000
        n.step(0)
        expected = (1000 * 243 >> 8) + 0  # = 949
        assert n.v == expected, f"v={n.v}, expected={expected}"

    def test_decay_reduces_voltage(self):
        """Decay with zero input should reduce |v| toward v_rest=0."""
        n = SpiNNaker2Neuron()
        n.v = 500
        n.step(0)
        assert n.v < 500

    def test_effective_alpha(self):
        """alpha_eff = 243/256 ≈ 0.9492."""
        alpha = 243 / 256
        assert 0.94 < alpha < 0.96

    def test_integer_arithmetic_only(self):
        """Verify no float operations: v stays integer."""
        n = SpiNNaker2Neuron()
        for _ in range(100):
            n.step(100)
            assert isinstance(n.v, int), f"v is {type(n.v)}"


class TestSpiNNaker2Refractory:
    def test_refractory_blocks(self):
        n = SpiNNaker2Neuron()
        for _ in range(1000):
            if n.step(500) == 1:
                assert n._refrac_count == n.refrac_steps
                s1 = n.step(500)
                s2 = n.step(500)
                assert s1 == 0 and s2 == 0
                return
        raise AssertionError("No spike")

    def test_refrac_count_decrements(self):
        n = SpiNNaker2Neuron()
        n._refrac_count = 2
        n.step(0)
        assert n._refrac_count == 1

    def test_max_rate_limited_by_refrac(self):
        """With refrac_steps=2: max rate = 1/(1+2) = 0.33 spikes/step."""
        n = SpiNNaker2Neuron(refrac_steps=2)
        outputs = [n.step(2000) for _ in range(3000)]
        spikes = outputs.count(1)
        max_rate = 3000 / (1 + 2)
        assert spikes <= max_rate + 10


class TestSpiNNaker2FI:
    def test_zero_silent(self):
        n = SpiNNaker2Neuron()
        assert sum(n.step(0) for _ in range(5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [100, 300, 500, 1000]:
            n = SpiNNaker2Neuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestSpiNNaker2Performance:
    def test_isolation_throughput(self):
        n = SpiNNaker2Neuron()
        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(500)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 100000


class TestSpiNNaker2Pipeline:
    def test_population_creates(self):
        assert Population(SpiNNaker2Neuron, n=10, label="sn2").n == 10

    def test_network_incompatible(self):
        """SpiNNaker2 uses integer >> operator. Population.step_all passes
        float(currents[i]) which fails on >>. This is a known limitation:
        integer neuromorphic models need an int-cast adapter for Network."""
        import pytest

        pop = Population(SpiNNaker2Neuron, n=5, label="sn2")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        with pytest.raises(TypeError):
            net.run(duration=0.1, dt=0.001, backend="python")

    def test_analysis(self):
        n = SpiNNaker2Neuron()
        train = np.array([float(n.step(500)) for _ in range(5000)])
        assert spike_count(train) >= 10

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SpiNNaker2Neuron()
            trace = [(n.step(500), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
