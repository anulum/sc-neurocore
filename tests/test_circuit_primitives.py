# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for circuit primitives

"""Tests for lateral inhibition and winner-take-all circuits."""

import numpy as np

from sc_neurocore.layers.circuit_primitives import LateralInhibition, WinnerTakeAll


class TestLateralInhibition:
    def test_suppresses_neighbors(self):
        li = LateralInhibition(n_neurons=5, inhibition_strength=0.5, radius=1)
        rates = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = li.apply(rates)
        assert result[2] == 1.0  # center untouched (no self-inhibition)
        assert result[1] == 0.0  # neighbor suppressed
        assert result[3] == 0.0  # neighbor suppressed
        assert result[0] == 0.0  # beyond radius

    def test_preserves_zero_input(self):
        li = LateralInhibition(n_neurons=10, inhibition_strength=0.5, radius=3)
        rates = np.zeros(10)
        result = li.apply(rates)
        np.testing.assert_allclose(result, 0.0)

    def test_circular_topology(self):
        li = LateralInhibition(n_neurons=6, inhibition_strength=0.5, radius=1)
        # Neuron 0 should inhibit neuron 5 (circular neighbor)
        rates = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        result = li.apply(rates)
        assert result[5] < 0.5  # inhibited by neuron 0

    def test_output_non_negative(self):
        li = LateralInhibition(n_neurons=10, inhibition_strength=1.0, radius=5)
        rates = np.random.rand(10)
        result = li.apply(rates)
        assert np.all(result >= 0.0)


class TestWinnerTakeAll:
    def test_single_winner(self):
        wta = WinnerTakeAll(n_neurons=5, k=1)
        rates = np.array([0.1, 0.8, 0.3, 0.5, 0.2])
        result = wta.apply(rates)
        assert result[1] == 0.8  # winner preserved
        assert np.count_nonzero(result) == 1

    def test_k_winners(self):
        wta = WinnerTakeAll(n_neurons=5, k=2)
        rates = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        result = wta.apply(rates)
        assert result[1] > 0  # 2nd place
        assert result[3] > 0  # 1st place
        assert np.count_nonzero(result) == 2

    def test_winners_indices(self):
        wta = WinnerTakeAll(n_neurons=5, k=2)
        rates = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        idx = wta.winners(rates)
        assert idx[0] == 3  # top winner
        assert idx[1] == 1  # 2nd winner

    def test_k_equals_n(self):
        wta = WinnerTakeAll(n_neurons=3, k=3)
        rates = np.array([0.5, 0.3, 0.7])
        result = wta.apply(rates)
        np.testing.assert_allclose(result, rates)

    def test_all_equal_rates(self):
        wta = WinnerTakeAll(n_neurons=4, k=1)
        rates = np.array([0.5, 0.5, 0.5, 0.5])
        result = wta.apply(rates)
        # All equal → WTA suppresses all (threshold = 0.5, rates <= threshold)
        assert np.count_nonzero(result) == 0
