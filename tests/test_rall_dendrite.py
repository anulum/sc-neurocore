# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rall dendrite model

"""Tests for Rall branching dendritic tree."""

import numpy as np

from sc_neurocore.layers.rall_dendrite import RallDendrite


class TestRallDendrite:
    def test_initial_zero(self):
        d = RallDendrite(n_branches=4, branch_length=3)
        assert d.soma_v == 0.0
        assert np.all(d.v == 0.0)

    def test_input_reaches_soma(self):
        """Distal input should propagate to soma over multiple steps."""
        d = RallDendrite(n_branches=2, branch_length=3, coupling=0.5)
        inputs = np.array([1.0, 0.0])
        for _ in range(20):
            v = d.step(inputs)
        assert v > 0

    def test_more_branches_more_input(self):
        """Input on all branches should produce larger soma voltage."""
        d1 = RallDendrite(n_branches=4, branch_length=2, coupling=0.5)
        d2 = RallDendrite(n_branches=4, branch_length=2, coupling=0.5)
        for _ in range(20):
            v1 = d1.step(np.array([1.0, 0.0, 0.0, 0.0]))
            v2 = d2.step(np.array([1.0, 1.0, 1.0, 1.0]))
        assert v2 > v1

    def test_decay_without_input(self):
        """Soma voltage should decay without input."""
        d = RallDendrite(n_branches=2, branch_length=2)
        d.step(np.array([5.0, 5.0]))
        d.step(np.array([5.0, 5.0]))
        v_peak = d.soma_v
        for _ in range(50):
            d.step(np.array([0.0, 0.0]))
        assert d.soma_v < v_peak

    def test_branch_voltages_shape(self):
        d = RallDendrite(n_branches=3, branch_length=4)
        assert d.branch_voltages.shape == (3, 4)

    def test_rall_attenuation(self):
        """Rall 3/2 rule: attenuation factors should sum to <= 1."""
        d = RallDendrite(n_branches=4)
        assert np.sum(d.attenuation) <= 1.0 + 1e-10

    def test_reset(self):
        d = RallDendrite(n_branches=2, branch_length=2)
        d.step(np.array([3.0, 3.0]))
        d.reset()
        assert d.soma_v == 0.0
        assert np.all(d.v == 0.0)

    def test_compartmental_gradient(self):
        """Distal compartments should have higher voltage than proximal when injected distally."""
        d = RallDendrite(n_branches=1, branch_length=5, coupling=0.3, tau=20.0)
        for _ in range(10):
            d.step(np.array([2.0]))
        bv = d.branch_voltages[0]
        # Distal tip (last) should have higher voltage than proximal (first)
        assert bv[-1] > bv[0]
