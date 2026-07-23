# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWinnerTakeAll from former test_circuit_primitives.py

"""Focused suite: TestWinnerTakeAll from former test_circuit_primitives.py."""

from __future__ import annotations

from tests.circuit_primitives_support import *  # noqa: F403

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
        # All equal → WTA still picks exactly k=1 winner (ties broken by index)
        assert np.count_nonzero(result) == 1
