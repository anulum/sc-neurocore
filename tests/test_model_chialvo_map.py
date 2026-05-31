# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ChialvoMapNeuron

"""Full pipeline test for ChialvoMapNeuron (Chialvo 1995).

2D discrete map neuron. x²·exp(y-x) dynamics. Intrinsically excitable
(spikes without input at default k=0.04)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestChialvoIsolation:
    def test_construction(self):
        n = ChialvoMapNeuron()
        assert n.x == 0.0
        assert n.y == 0.0

    def test_step_returns_binary(self):
        n = ChialvoMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_intrinsic_spiking(self):
        """Model spikes without input (k=0.04 provides excitability)."""
        n = ChialvoMapNeuron()
        spikes = sum(n.step(0.0) for _ in range(5000))
        assert spikes > 0, "no intrinsic spiking"

    def test_state_finite(self):
        n = ChialvoMapNeuron()
        for _ in range(10000):
            n.step(0.02)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_safe_exp_prevents_overflow(self):
        """Extreme y-x should not cause overflow (safe_exp used)."""
        n = ChialvoMapNeuron()
        n.y = 1000.0
        n.x = 0.0
        result = n.step(0.0)
        assert result in (0, 1)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_reset(self):
        n = ChialvoMapNeuron()
        for _ in range(100):
            n.step(0.02)
        n.reset()
        assert n.x == 0.0
        assert n.y == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("x", np.nan),
            ("y", np.inf),
            ("a", np.nan),
            ("b", np.inf),
            ("c", np.nan),
            ("k", np.inf),
            ("x_threshold", np.nan),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            ChialvoMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = ChialvoMapNeuron()
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.x, n.y) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = ChialvoMapNeuron()
        n.y = np.inf
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(0.0)
        assert (n.x, n.y) == before

    def test_rejects_quadratic_overflow_before_state_mutation(self):
        n = ChialvoMapNeuron(x=1.0e308, y=0.0)
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="quadratic|candidate"):
            n.step(0.0)
        assert (n.x, n.y) == before


class TestChialvoNetwork:
    def test_population(self):
        pop = Population(ChialvoMapNeuron, n=10, label="chialvo")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(ChialvoMapNeuron, n=20, label="chialvo")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.1, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(ChialvoMapNeuron, n=10, label="chialvo")
        proj = Projection(pop, pop, weight=0.01, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.05, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestChialvoAnalysis:
    def _get_train(self):
        n = ChialvoMapNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(0.0)
        return train

    def test_firing_rate(self):
        train = self._get_train()
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_train()
        assert spike_count(train) > 0

    def test_isi(self):
        train = self._get_train()
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
