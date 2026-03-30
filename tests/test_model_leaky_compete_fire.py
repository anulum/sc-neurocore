# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LeakyCompeteFireNeuron

"""Full pipeline test for LeakyCompeteFireNeuron (Oster et al. 2009).

Winner-take-all with lateral inhibition. Multi-unit: returns list[int]."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.leaky_compete_fire import LeakyCompeteFireNeuron


class TestLCFIsolation:
    def test_construction(self):
        n = LeakyCompeteFireNeuron()
        assert n.n_units == 4
        assert len(n.v) == 4

    def test_step_returns_list(self):
        n = LeakyCompeteFireNeuron()
        s = n.step([0.0, 0.0, 0.0, 0.0])
        assert isinstance(s, list)
        assert len(s) == 4
        assert all(x in (0, 1) for x in s)

    def test_scalar_input_broadcast(self):
        """Scalar current should be broadcast to all units."""
        n = LeakyCompeteFireNeuron()
        s = n.step(0.0)
        assert len(s) == 4

    def test_winner_take_all(self):
        """Strongest input should dominate — others suppressed."""
        n = LeakyCompeteFireNeuron()
        totals = [0] * 4
        for _ in range(1000):
            s = n.step([2.0, 1.0, 0.5, 0.2])
            for i in range(4):
                totals[i] += s[i]
        assert totals[0] > 50
        assert totals[1] + totals[2] + totals[3] < totals[0]

    def test_lateral_inhibition(self):
        """When unit 0 spikes, others' voltage should decrease."""
        n = LeakyCompeteFireNeuron()
        n.v = [1.5, 0.8, 0.8, 0.8]
        s = n.step([0.0, 0.0, 0.0, 0.0])
        if s[0] == 1:
            assert all(n.v[i] <= 0.8 for i in range(1, 4))

    def test_no_negative_voltage(self):
        """Voltage should be clamped to >= 0 after inhibition."""
        n = LeakyCompeteFireNeuron()
        n.v = [1.5, 0.1, 0.1, 0.1]
        n.step([0.0, 0.0, 0.0, 0.0])
        assert all(v >= 0.0 for v in n.v)

    def test_equal_inputs(self):
        """Equal inputs — all units spike similarly."""
        n = LeakyCompeteFireNeuron()
        totals = [0] * 4
        for _ in range(2000):
            s = n.step([2.0, 2.0, 2.0, 2.0])
            for i in range(4):
                totals[i] += s[i]
        assert all(t > 10 for t in totals)

    def test_custom_n_units(self):
        n = LeakyCompeteFireNeuron(n_units=8)
        assert len(n.v) == 8
        s = n.step([1.0] * 8)
        assert len(s) == 8

    def test_numerical_stability(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(5000):
            n.step([2.0, 1.0, 0.5, 0.2])
        assert all(np.isfinite(v) for v in n.v)

    def test_reset(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(500):
            n.step([2.0, 1.0, 0.5, 0.2])
        n.reset()
        assert all(v == 0.0 for v in n.v)

    def test_deterministic(self):
        n1 = LeakyCompeteFireNeuron()
        n2 = LeakyCompeteFireNeuron()
        for _ in range(200):
            assert n1.step([2.0, 1.0, 0.5, 0.2]) == n2.step([2.0, 1.0, 0.5, 0.2])


class TestLCFNetwork:
    def test_standalone_multi_unit(self):
        """LCF is multi-unit — v is list, not float. Standard Population
        does not support list-valued v. LCF is used standalone."""
        n = LeakyCompeteFireNeuron(n_units=8)
        for _ in range(100):
            n.step([1.0] * 8)
        assert len(n.v) == 8
