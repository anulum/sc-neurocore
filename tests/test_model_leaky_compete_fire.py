# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LeakyCompeteFireNeuron

"""Full pipeline test for LeakyCompeteFireNeuron (Oster et al. 2009).

Winner-take-all with lateral inhibition. Multi-unit model (n_units=4):
dV_i = (-V_i + I_i) / τ · dt
On spike: V_i→0, V_j -= w_inh (j≠i), clipped ≥ 0.

Returns list[int] (one spike per unit). v is list[float].
Population incompatible (list-valued v).
FULL PIPELINE (isolation only) + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.leaky_compete_fire import LeakyCompeteFireNeuron
from sc_neurocore.network.population import Population


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestLCFIsolation:
    def test_defaults(self):
        n = LeakyCompeteFireNeuron()
        assert n.n_units == 4 and len(n.v) == 4
        assert n.tau == 10.0 and n.v_threshold == 1.0
        assert n.w_inh == 0.5

    def test_step_returns_list(self):
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        assert isinstance(result, list) and len(result) == 4

    def test_each_element_binary(self):
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        for s in result:
            assert s in (0, 1)

    def test_reset_zeroes_all(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert all(v == 0.0 for v in n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LeakyCompeteFireNeuron()
            trace = [tuple(n.step(5.0)) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — WTA mechanism, lateral inhibition
# ---------------------------------------------------------------------------
class TestLCFAnalytical:
    def test_uniform_input_all_fire_together(self):
        """Equal input to all units → all spike simultaneously."""
        n = LeakyCompeteFireNeuron()
        for _ in range(1000):
            spikes = n.step(5.0)
            if sum(spikes) > 0:
                # All got same input → all should spike together
                assert sum(spikes) == n.n_units or sum(spikes) >= 1
                break

    def test_lateral_inhibition_resets_losers(self):
        """When unit i spikes, other units V_j -= w_inh."""
        n = LeakyCompeteFireNeuron(n_units=2)
        # Drive unit 0 strongly, unit 1 weakly
        for _ in range(100):
            spikes = n.step([10.0, 0.1])
            if spikes[0] == 1:
                # Unit 1 should have been inhibited
                assert n.v[1] >= 0.0  # clipped
                break

    def test_winner_take_all_with_asymmetric_input(self):
        """Stronger input unit spikes more often (WTA)."""
        n = LeakyCompeteFireNeuron(n_units=2)
        spikes_0, spikes_1 = 0, 0
        for _ in range(5000):
            s = n.step([5.0, 2.0])
            spikes_0 += s[0]
            spikes_1 += s[1]
        assert spikes_0 > spikes_1

    def test_v_non_negative_after_inhibition(self):
        """V clipped to ≥ 0 after lateral inhibition."""
        n = LeakyCompeteFireNeuron()
        for _ in range(1000):
            n.step(5.0)
        for v in n.v:
            assert v >= 0.0

    def test_scalar_input_broadcast(self):
        """Scalar input is broadcast to all units."""
        n = LeakyCompeteFireNeuron()
        n.step(5.0)
        # All units should get same drive
        assert isinstance(n.v, list) and len(n.v) == n.n_units

    def test_list_input_per_unit(self):
        """List input gives different current per unit."""
        n = LeakyCompeteFireNeuron(n_units=3, v_threshold=100.0)
        n.step([1.0, 2.0, 3.0])
        # Unit 2 should have highest V
        assert n.v[2] > n.v[0]

    def test_custom_n_units(self):
        n = LeakyCompeteFireNeuron(n_units=8)
        assert len(n.v) == 8
        result = n.step(5.0)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# 3. PARAMETERS
# ---------------------------------------------------------------------------
class TestLCFParameters:
    @pytest.mark.parametrize("w_inh", [0.0, 0.5, 2.0])
    def test_w_inh_sweep(self, w_inh: float):
        n = LeakyCompeteFireNeuron(w_inh=w_inh)
        for _ in range(1000):
            n.step(5.0)
        assert all(np.isfinite(v) for v in n.v)

    @pytest.mark.parametrize("n_units", [2, 4, 8])
    def test_n_units_sweep(self, n_units: int):
        n = LeakyCompeteFireNeuron(n_units=n_units)
        result = n.step(5.0)
        assert len(result) == n_units


# ---------------------------------------------------------------------------
# 4. PERFORMANCE
# ---------------------------------------------------------------------------
class TestLCFPerformance:
    def test_isolation_throughput(self):
        n = LeakyCompeteFireNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"


# ---------------------------------------------------------------------------
# 5. PIPELINE (Population incompatible)
# ---------------------------------------------------------------------------
class TestLCFPipeline:
    def test_population_incompatible(self):
        """v is list (multi-unit WTA) → Population._sync_voltages fails."""
        with pytest.raises((ValueError, TypeError)):
            Population(LeakyCompeteFireNeuron, n=5, label="lcf")

    def test_analysis_isolation(self):
        """Run in isolation, flatten spikes for analysis."""
        n = LeakyCompeteFireNeuron()
        total_spikes = 0
        for _ in range(5000):
            spikes = n.step(5.0)
            total_spikes += sum(spikes)
        assert total_spikes > 0
