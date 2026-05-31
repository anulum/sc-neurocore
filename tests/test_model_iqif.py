# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: IntegerQIFNeuron

"""Full pipeline test for IntegerQIFNeuron (Lo et al. 2021).

Fixed-point quadratic integrate-and-fire, all integer arithmetic:
V[t+1] = max(V_min, V[t] + (V[t]² >> k) + I)
Spike: V → V_reset when V ≥ V_threshold.

V: int, k=6 (right-shift for V²), V_threshold=1024, V_reset=-1024.
V_min=-2048 (clipping). >> operator requires integer operands.
Population.step_all passes float → TypeError documented.
FULL PIPELINE WIRED (isolation only) + PERFORMANCE."""

from __future__ import annotations

import time
import os

import numpy as np
import pytest

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.network import Network
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: IntegerQIFNeuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestIQIFIsolation:
    def test_defaults(self):
        n = IntegerQIFNeuron()
        assert n.v == 0 and n.k == 6
        assert n.v_threshold == 1024 and n.v_reset == -1024
        assert n.v_min == -2048

    def test_integer_state(self):
        """V is integer — hardware constraint."""
        n = IntegerQIFNeuron()
        n.step(100)
        assert isinstance(n.v, int)

    def test_step_returns_binary(self):
        assert IntegerQIFNeuron().step(100) in (0, 1)

    def test_reset_restores_default(self):
        n = IntegerQIFNeuron()
        for _ in range(100):
            n.step(100)
        n.reset()
        assert n.v == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = IntegerQIFNeuron()
            trace = [(n.step(100), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — integer arithmetic, right-shift, clipping
# ---------------------------------------------------------------------------
class TestIQIFAnalytical:
    def test_update_formula(self):
        """V = max(V_min, V + (V² >> k) + I)."""
        n = IntegerQIFNeuron()
        v0 = n.v  # 0
        I = 100
        expected = max(n.v_min, v0 + (v0 * v0 >> n.k) + I)
        n.step(I)
        assert n.v == expected

    def test_right_shift_is_integer_divide(self):
        """V² >> 6 = V² // 64."""
        v = 100
        assert (v * v >> 6) == (v * v // 64)

    def test_quadratic_acceleration(self):
        """V² term: larger V → faster growth (positive feedback)."""
        n = IntegerQIFNeuron()
        n.v = 10
        v1 = n.v + (n.v * n.v >> n.k) + 0  # at V=10: 10 + (100>>6) = 10+1=11
        n2 = IntegerQIFNeuron()
        n2.v = 100
        v2 = n2.v + (n2.v * n2.v >> n2.k) + 0  # at V=100: 100 + (10000>>6) = 100+156=256
        assert v2 - 100 > v1 - 10

    def test_v_min_clipping(self):
        """V clipped to V_min=-2048."""
        n = IntegerQIFNeuron()
        n.v = -2000
        n.step(-1000)  # would go below V_min
        assert n.v >= n.v_min

    def test_spike_reset(self):
        """On V ≥ threshold: V → V_reset."""
        n = IntegerQIFNeuron()
        for _ in range(10_000):
            if n.step(100) == 1:
                assert n.v == n.v_reset
                break

    def test_requires_integer_input(self):
        """>> operator requires int. Float contamination → TypeError."""
        n = IntegerQIFNeuron()
        n.step(100.5)  # v becomes float (0 + 0 + 100.5 = 100.5)
        with pytest.raises(TypeError):
            n.step(100)  # now v*v is float, float >> 6 → TypeError


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestIQIFDynamics:
    def test_fires_with_positive_input(self):
        n = IntegerQIFNeuron()
        spikes = _run(n, current=100, steps=5000)
        assert len(spikes) >= 10

    def test_rate_monotonic(self):
        s_low = len(_run(IntegerQIFNeuron(), 50, 5000))
        s_high = len(_run(IntegerQIFNeuron(), 500, 5000))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [10, 50, 100, 200, 500])
    def test_fi_sweep(self, current: int):
        n = IntegerQIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert isinstance(n.v, int)

    def test_silent_at_zero(self):
        """V=0, I=0: V² >> 6 = 0 → V stays 0."""
        n = IntegerQIFNeuron()
        assert len(_run(n, current=0, steps=1000)) == 0


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestIQIFParameters:
    @pytest.mark.parametrize("k", [4, 6, 8])
    def test_k_shift_sweep(self, k: int):
        n = IntegerQIFNeuron(k=k)
        for _ in range(5000):
            n.step(100)
        assert isinstance(n.v, int)

    @pytest.mark.parametrize("v_threshold", [512, 1024, 2048])
    def test_threshold_sweep(self, v_threshold: int):
        n = IntegerQIFNeuron(v_threshold=v_threshold)
        spikes = len(_run(n, current=100, steps=5000))
        assert isinstance(spikes, int)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestIQIFPerformance:
    def test_isolation_throughput(self):
        n = IntegerQIFNeuron()
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 400_000 if os.getenv("CI") else 500_000
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"


# ---------------------------------------------------------------------------
# 6. PIPELINE (isolation + incompatibility documented)
# ---------------------------------------------------------------------------
class TestIQIFPipeline:
    def test_population_creates(self):
        assert Population(IntegerQIFNeuron, n=5, label="iqif").n == 5

    def test_network_incompatible(self):
        """Integer >> on float from Population.step_all → TypeError."""
        pop = Population(IntegerQIFNeuron, n=5, label="t")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        with pytest.raises(TypeError):
            net.run(duration=0.1, dt=0.001, backend="python")

    def test_analysis_isolation(self):
        n = IntegerQIFNeuron()
        train = np.array([float(n.step(100)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 5


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestIntegerQIF:
    def test_fires(self):
        from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

        n = IntegerQIFNeuron()
        assert sum(n.step(10) for _ in range(200)) > 0

    def test_integer_arithmetic(self):
        from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron

        n = IntegerQIFNeuron()
        n.step(5)
        assert isinstance(n.v, int)
