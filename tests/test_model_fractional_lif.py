# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: FractionalLIFNeuron

"""Full pipeline test for FractionalLIFNeuron (Lundstrom et al. 2008).

Grünwald-Letnikov fractional derivative: D^α v = -(v-v_rest) + R·I.
α < 1 introduces power-law memory (not exponential decay)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.fractional_lif import FractionalLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestFracLIFIsolation:
    def test_construction(self):
        n = FractionalLIFNeuron()
        assert n.v == 0.0
        assert n.alpha == 0.8

    def test_step_returns_binary(self):
        assert FractionalLIFNeuron().step(0.0) in (0, 1)

    def test_silent_at_zero(self):
        n = FractionalLIFNeuron()
        assert sum(n.step(0.0) for _ in range(1000)) == 0

    def test_spikes_under_drive(self):
        n = FractionalLIFNeuron()
        assert sum(n.step(0.5) for _ in range(1000)) > 50

    def test_alpha_effect(self):
        """Lower alpha (more memory) should produce fewer spikes."""
        n_low = FractionalLIFNeuron(alpha=0.5)
        n_high = FractionalLIFNeuron(alpha=0.9)
        s_low = sum(n_low.step(0.3) for _ in range(3000))
        s_high = sum(n_high.step(0.3) for _ in range(3000))
        assert s_high > s_low

    def test_history_maintained(self):
        """Internal GL history buffer should be populated after stepping."""
        n = FractionalLIFNeuron()
        for _ in range(200):
            n.step(0.3)
        assert len(n._history) == n._max_history

    def test_gl_coefficients(self):
        """GL coefficients should be computed and non-trivial."""
        n = FractionalLIFNeuron()
        assert len(n._gl_coeffs) == n._max_history
        assert n._gl_coeffs[0] == 1.0
        assert n._gl_coeffs[1] != 0.0

    def test_numerical_stability(self):
        for I in [0.0, 0.3, 0.5, 1.0]:
            n = FractionalLIFNeuron()
            for _ in range(3000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"

    def test_reset(self):
        n = FractionalLIFNeuron()
        for _ in range(500):
            n.step(0.5)
        n.reset()
        assert n.v == n.v_rest
        assert all(h == 0.0 for h in n._history)

    def test_custom_max_history(self):
        n = FractionalLIFNeuron(_max_history=50)
        assert len(n._gl_coeffs) == 50
        for _ in range(200):
            n.step(0.3)
        assert len(n._history) == 50

    def test_alpha_one_reduces_to_standard(self):
        """α=1.0 should behave like standard LIF (high spike rate)."""
        n = FractionalLIFNeuron(alpha=1.0)
        s = sum(n.step(0.5) for _ in range(1000))
        assert s > 100


class TestFracLIFNetwork:
    def test_population(self):
        assert Population(FractionalLIFNeuron, n=10, label="flif").n == 10

    def test_network_spikes(self):
        pop = Population(FractionalLIFNeuron, n=10, label="flif")
        drive = PoissonInput(n=10, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestFracLIFAnalysis:
    def test_spike_count(self):
        n = FractionalLIFNeuron()
        train = np.zeros(2000, dtype=np.int8)
        for t in range(2000):
            train[t] = n.step(0.5)
        assert spike_count(train) > 100
