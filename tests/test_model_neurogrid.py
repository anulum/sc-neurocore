# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NeuroGridNeuron

"""Full pipeline test for NeuroGridNeuron (Boahen 2014).

2-compartment: dendrite (passive) + soma (EIF). Analog neuromorphic chip."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestNGIsolation:
    def test_construction(self):
        n = NeuroGridNeuron()
        assert n.v_s == -65.0
        assert n.v_d == -65.0

    def test_step_returns_binary(self):
        assert NeuroGridNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = NeuroGridNeuron()
        assert sum(n.step(20.0) for _ in range(3000)) == 0

    def test_spikes_under_drive(self):
        n = NeuroGridNeuron()
        assert sum(n.step(100.0) for _ in range(5000)) > 5

    def test_two_compartments(self):
        n = NeuroGridNeuron()
        for _ in range(500):
            n.step(50.0)
        assert n.v_d != n.v_s

    def test_dendritic_integration(self):
        """Dendrite should accumulate input."""
        n = NeuroGridNeuron()
        for _ in range(200):
            n.step(50.0)
        assert n.v_d > n.v_rest

    def test_numerical_stability(self):
        for I in [0.0, 50.0, 100.0]:
            n = NeuroGridNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v_s)
            assert np.isfinite(n.v_d)

    def test_reset(self):
        n = NeuroGridNeuron()
        for _ in range(2000):
            n.step(100.0)
        n.reset()
        assert n.v_s == -65.0
        assert n.v_d == -65.0

    def test_deterministic(self):
        n1 = NeuroGridNeuron()
        n2 = NeuroGridNeuron()
        for _ in range(300):
            assert n1.step(80.0) == n2.step(80.0)


class TestNGNetwork:
    def test_population(self):
        assert Population(NeuroGridNeuron, n=5, label="ng").n == 5


class TestNGAnalysis:
    def test_spike_count(self):
        n = NeuroGridNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(100.0)
        assert spike_count(train) > 5
