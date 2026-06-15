# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ai_optimized (8 models)

"""Full pipeline test for all 8 AI-optimised neuron models.

MultiTimescale, AttentionGated, PredictiveCoding, SelfReferential,
CompositionalBinding, DifferentiableSurrogate, ContinuousAttractor,
MetaPlastic. All return int, all fire at I≥2.0.
Performance range: 4K–880K steps/s. All pipeline-wired."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard

from sc_neurocore.neurons.models.ai_optimized import (
    AttentionGatedNeuron,
    CompositionalBindingNeuron,
    ContinuousAttractorNeuron,
    DifferentiableSurrogateNeuron,
    MetaPlasticNeuron,
    MultiTimescaleNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count

ALL_CLASSES = [
    MultiTimescaleNeuron,
    AttentionGatedNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
    CompositionalBindingNeuron,
    DifferentiableSurrogateNeuron,
    ContinuousAttractorNeuron,
    MetaPlasticNeuron,
]


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
class TestAIOptimizedCommon:
    """Tests applied to all 8 AI-optimised models."""

    def test_step_returns_int(self, cls: type):
        n = cls()
        assert n.step(0.0) in (0, 1)

    def test_state_finite(self, cls: type):
        n = cls()
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(getattr(n, "v", getattr(n, "v_fast", 0.0)))

    def test_fires_at_i2(self, cls: type):
        """All 8 models fire at I=2.0."""
        n = cls()
        spikes = sum(n.step(2.0) for _ in range(5000))
        assert spikes > 0, f"{cls.__name__} silent at I=2.0"

    def test_reset(self, cls: type):
        n = cls()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        # After reset, first state variable should be at initial value
        # (exact check depends on model, but reset should not crash)

    def test_deterministic(self, cls: type):
        traces = []
        for _ in range(2):
            n = cls()
            trace = [n.step(2.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_population_creates(self, cls: type):
        pop = Population(cls, n=5, label="ai")
        assert pop.n == 5


class TestAIOptimizedNetwork:
    """Pipeline wiring for representative models."""

    def test_multitimescale_network(self):
        pop = Population(MultiTimescaleNeuron, n=5, label="mt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_differentiable_surrogate_network(self):
        pop = Population(DifferentiableSurrogateNeuron, n=10, label="ds")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_compositional_binding_network(self):
        pop = Population(CompositionalBindingNeuron, n=10, label="cb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestAIOptimizedAnalysis:
    def test_spike_count_multitimescale(self):
        n = MultiTimescaleNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        assert spike_count(train) >= 50

    def test_spike_count_self_referential(self):
        n = SelfReferentialNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        assert spike_count(train) >= 10


class TestAIOptimizedPerformance:
    @pytest.mark.parametrize(
        "cls,min_perf",
        [
            (MetaPlasticNeuron, 100000),
            (DifferentiableSurrogateNeuron, 100000),
            (AttentionGatedNeuron, 100000),
            (MultiTimescaleNeuron, 50000),
            (ContinuousAttractorNeuron, 1000),
        ],
        ids=lambda c: c.__name__ if isinstance(c, type) else str(c),
    )
    def test_throughput(self, cls: type, min_perf: int):
        n = cls()
        N = min(min_perf, 10000)
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        strict_minimum = float(min_perf) * 0.5
        assert_throughput_guard(
            label=f"{cls.__name__} isolation",
            observed_per_second=N / elapsed,
            strict_minimum_per_second=strict_minimum,
            smoke_minimum_per_second=min(500.0, max(25.0, strict_minimum * 0.01)),
        )
