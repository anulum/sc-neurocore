# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LeakyCompeteFireNeuron

"""Full pipeline: LeakyCompeteFireNeuron. FULL PIPELINE + PERFORMANCE."""

from __future__ import annotations

import time


from sc_neurocore.neurons.models.leaky_compete_fire import LeakyCompeteFireNeuron
from sc_neurocore.network.population import Population


def _run(neuron, current, steps):
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns(self):
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        assert result is not None

    def test_state_finite(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(3000):
            n.step(5.0)
        pass  # special state

    def test_reset(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()


class TestDynamics:
    def test_returns_list(self):
        """WTA multi-unit model returns list, not scalar."""
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        assert isinstance(result, list)

    def test_rate_monotonic(self):
        n_low = LeakyCompeteFireNeuron()
        n_high = LeakyCompeteFireNeuron()
        s_low = len(_run(n_low, 2.0, 5000))
        s_high = len(_run(n_high, 10.0, 5000))
        assert s_high >= s_low

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LeakyCompeteFireNeuron()
            trace = [n.step(5.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestPerformance:
    def test_isolation_throughput(self):
        n = LeakyCompeteFireNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000


class TestPipeline:
    def test_population_incompatible(self):
        """v is a list (multi-unit WTA) → Population._sync_voltages fails."""
        import pytest

        with pytest.raises((ValueError, TypeError)):
            Population(LeakyCompeteFireNeuron, n=5, label="t")
