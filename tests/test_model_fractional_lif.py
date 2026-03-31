# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: FractionalLIFNeuron

"""Full pipeline test for FractionalLIFNeuron (Lundstrom et al. 2008).

Grünwald-Letnikov fractional derivative: D^α v = -(v-v_rest) + R·I.
α<1 introduces memory (power-law decay). History buffer of 100 steps.
GL coefficients: c[k] = c[k-1]·(k-1-α)/k. Performance: ~29K steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.fractional_lif import FractionalLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: FractionalLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestFLIFIsolation:
    def test_defaults(self):
        n = FractionalLIFNeuron()
        assert n.v == 0.0 and n.alpha == 0.8 and n.v_threshold == 1.0
        assert n._max_history == 100

    def test_step_returns_binary(self):
        assert FractionalLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = FractionalLIFNeuron()
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = FractionalLIFNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == n.v_rest
        assert len(n._history) == n._max_history


class TestFLIFGLCoefficients:
    """Grünwald-Letnikov coefficients: c[0]=1, c[k] = c[k-1]·(k-1-α)/k."""

    def test_first_coefficient_is_one(self):
        n = FractionalLIFNeuron()
        assert n._gl_coeffs[0] == 1.0

    def test_gl_recurrence(self):
        """c[k] = c[k-1] · (k-1-alpha) / k."""
        n = FractionalLIFNeuron(alpha=0.8)
        for k in range(1, 10):
            expected = n._gl_coeffs[k - 1] * (k - 1 - 0.8) / k
            assert abs(n._gl_coeffs[k] - expected) < 1e-12

    def test_alpha_1_reduces_to_lif(self):
        """At α=1: GL coeffs → [1, 0, 0, ...] (standard derivative)."""
        n = FractionalLIFNeuron(alpha=1.0)
        assert n._gl_coeffs[0] == 1.0
        # c[1] = 1 * (0 - 1) / 1 = -1
        assert abs(n._gl_coeffs[1] - (-1.0)) < 1e-12

    def test_alpha_affects_memory_depth(self):
        """Lower α → slower coefficient decay → longer effective memory."""
        n_low = FractionalLIFNeuron(alpha=0.5)
        n_high = FractionalLIFNeuron(alpha=0.9)
        # At k=50: low alpha should have larger |coeff|
        assert abs(n_low._gl_coeffs[50]) > abs(n_high._gl_coeffs[50])

    def test_history_buffer_length(self):
        n = FractionalLIFNeuron()
        for _ in range(200):
            n.step(5.0)
        assert len(n._history) == n._max_history


class TestFLIFDynamics:
    def test_fires_at_sufficient_current(self):
        n = FractionalLIFNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 100

    def test_zero_input_silent(self):
        n = FractionalLIFNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_rate_increases_with_current(self):
        n5 = FractionalLIFNeuron()
        n10 = FractionalLIFNeuron()
        s5 = len(_run(n5, current=5.0, steps=5000))
        s10 = len(_run(n10, current=10.0, steps=5000))
        assert s10 >= s5

    def test_alpha_affects_dynamics(self):
        """Lower α → more memory → different firing pattern."""
        n_low = FractionalLIFNeuron(alpha=0.5)
        n_high = FractionalLIFNeuron(alpha=0.95)
        s_low = len(_run(n_low, current=0.5, steps=5000))
        s_high = len(_run(n_high, current=0.5, steps=5000))
        assert s_low != s_high


class TestFLIFParameters:
    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
    def test_alpha_variations(self, alpha: float):
        n = FractionalLIFNeuron(alpha=alpha)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FractionalLIFNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestFLIFPerformance:
    def test_isolation_throughput(self):
        n = FractionalLIFNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000

    def test_network_throughput(self):
        pop = Population(FractionalLIFNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 20 * 500 / elapsed > 1000


class TestFLIFPipeline:
    def test_population(self):
        assert Population(FractionalLIFNeuron, n=10, label="flif").n == 10

    def test_network_spikes(self):
        pop = Population(FractionalLIFNeuron, n=10, label="flif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(FractionalLIFNeuron, n=5, label="src")
        tgt = Population(FractionalLIFNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=3.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = FractionalLIFNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
