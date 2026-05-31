# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AkidaNeuron

"""Full pipeline test for AkidaNeuron (BrainChip Akida 2021).

Event-domain rank-order integrate-and-fire neuron:
V += int(weight · modulation^rank)
rank increments per non-zero input event.

Key properties:
- Integer arithmetic (V: int, weight: int)
- Rank-order coding: earlier events weighted more (modulation=0.75)
- Single-spike model: fires AT MOST ONCE (_spiked flag)
- No leak between events
- No reset after spike (just flags _spiked)

Performance: ~1.1M steps/s (integer arithmetic).
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time
import os

import numpy as np
import pytest

from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestAkidaIsolation:
    def test_defaults(self):
        n = AkidaNeuron()
        assert n.v == 0 and n.threshold == 100
        assert n.modulation == 0.75
        assert n._rank == 0 and n._spiked is False

    def test_step_returns_binary(self):
        assert AkidaNeuron().step(0) in (0, 1)

    def test_integer_voltage(self):
        """V is integer — neuromorphic hardware constraint."""
        n = AkidaNeuron()
        n.step(50)
        assert isinstance(n.v, int)

    def test_reset_restores_defaults(self):
        n = AkidaNeuron()
        for _ in range(10):
            n.step(50)
        n.reset()
        assert n.v == 0 and n._rank == 0 and n._spiked is False

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AkidaNeuron()
            trace = [(n.step(50), n.v) for _ in range(20)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — rank-order decay, integer scaling, single-spike
# ---------------------------------------------------------------------------
class TestAkidaAnalytical:
    def test_rank_order_decay_formula(self):
        """V += int(weight · modulation^rank). Rank increments per event."""
        n = AkidaNeuron()
        w = 50
        # Rank 0: scaled = int(50 * 0.75^0) = 50
        n.step(w)
        assert n.v == 50 and n._rank == 1
        # Rank 1: scaled = int(50 * 0.75^1) = int(37.5) = 37
        n.step(w)
        assert n.v == 50 + 37 and n._rank == 2
        # Rank 2: scaled = int(50 * 0.75^2) = int(28.125) = 28
        n.step(w)
        assert n.v == 50 + 37 + 28 and n._rank == 3

    def test_modulation_decay_sequence(self):
        """modulation^rank: 1.0, 0.75, 0.5625, 0.4219, ..."""
        m = 0.75
        expected = [m**k for k in range(5)]
        assert abs(expected[0] - 1.0) < 1e-12
        assert abs(expected[1] - 0.75) < 1e-12
        assert abs(expected[2] - 0.5625) < 1e-12

    def test_zero_weight_no_integration(self):
        """weight=0 → no integration, rank does not increment."""
        n = AkidaNeuron()
        n.step(0)
        assert n.v == 0 and n._rank == 0

    def test_rank_only_increments_on_nonzero(self):
        """Rank increments only when weight != 0."""
        n = AkidaNeuron()
        n.step(50)  # rank → 1
        n.step(0)  # rank stays 1
        n.step(50)  # rank → 2
        assert n._rank == 2

    def test_single_spike_only(self):
        """Once _spiked=True, neuron never fires again."""
        n = AkidaNeuron(threshold=50)
        # First spike
        n.step(60)
        assert n._spiked is True
        # All subsequent steps return 0
        for _ in range(100):
            assert n.step(60) == 0

    def test_spike_at_threshold(self):
        """Spike when V >= threshold and not already spiked."""
        n = AkidaNeuron(threshold=100)
        # Feed until threshold
        n.step(100)  # V = 100 → spike
        assert n._spiked is True

    def test_no_leak(self):
        """No leak between events — V persists."""
        n = AkidaNeuron()
        n.step(50)
        v_after = n.v
        n.step(0)  # zero input
        assert n.v == v_after  # unchanged

    def test_integer_truncation(self):
        """int() truncates toward zero."""
        n = AkidaNeuron()
        # weight=1, rank=1: int(1 * 0.75) = int(0.75) = 0
        n.step(10)  # rank=0: int(10*1.0) = 10
        v1 = n.v
        n.step(1)  # rank=1: int(1*0.75) = 0
        assert n.v == v1  # no change (scaled=0)


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestAkidaDynamics:
    def test_fires_with_large_weight(self):
        n = AkidaNeuron()
        assert n.step(100) == 1

    def test_accumulation_to_threshold(self):
        """Multiple small weights accumulate to threshold."""
        n = AkidaNeuron(threshold=100)
        spikes = 0
        for _ in range(50):
            spikes += n.step(30)
        assert spikes == 1  # fires once

    def test_never_fires_with_tiny_input(self):
        """Weight too small → int truncation → V never reaches threshold."""
        n = AkidaNeuron(threshold=100)
        # weight=1: ranks 0→0=1, 1→0, 2→0, ... V maxes at 1
        for _ in range(1000):
            n.step(1)
        assert n._spiked is False

    @pytest.mark.parametrize("weight", [20, 50, 100, 200])
    def test_weight_sweep(self, weight: int):
        n = AkidaNeuron()
        for _ in range(100):
            n.step(weight)
        assert isinstance(n.v, int)


# ---------------------------------------------------------------------------
# 4. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestAkidaParameters:
    @pytest.mark.parametrize("threshold", [50, 100, 200])
    def test_threshold_sweep(self, threshold: int):
        n = AkidaNeuron(threshold=threshold)
        for _ in range(100):
            n.step(50)
        # Should have spiked if threshold low enough
        if threshold <= 50:
            assert n._spiked is True

    @pytest.mark.parametrize("modulation", [0.5, 0.75, 0.9])
    def test_modulation_sweep(self, modulation: float):
        n = AkidaNeuron(modulation=modulation)
        for _ in range(20):
            n.step(50)
        assert isinstance(n.v, int)

    def test_higher_modulation_more_accumulation(self):
        """Higher modulation → slower decay → more total integration."""
        n_low = AkidaNeuron(modulation=0.5, threshold=10000)
        n_high = AkidaNeuron(modulation=0.9, threshold=10000)
        for _ in range(20):
            n_low.step(50)
            n_high.step(50)
        assert n_high.v >= n_low.v


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestAkidaPerformance:
    def test_isolation_throughput(self):
        n = AkidaNeuron(threshold=1_000_000)  # prevent spike to measure raw perf
        N = 500_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 400_000 if os.getenv("CI") else 500_000
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"

    def test_network_throughput(self):
        pop = Population(AkidaNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestAkidaPipeline:
    def test_population(self):
        assert Population(AkidaNeuron, n=10, label="akida").n == 10

    def test_projection_wiring(self):
        src = Population(AkidaNeuron, n=5, label="src")
        tgt = Population(AkidaNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(AkidaNeuron, n=10, label="akida")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # Single-spike model: at most n spikes total
        assert mon.count <= 10

    def test_analysis(self):
        n = AkidaNeuron()
        train = np.array([float(n.step(100)) for _ in range(100)])
        sc = spike_count(train)
        # Single spike model
        assert sc <= 1
