# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGapJunctionPopulation from former test_gap_junction.py

"""Focused suite: TestGapJunctionPopulation from former test_gap_junction.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.synapses.gap_junction import (
    GapJunction,
)


class TestGapJunctionPopulation:
    """Gap-junction currents across a population."""

    def test_current_matrix_two_neurons(self) -> None:
        """Two coupled neurons exchange opposite currents."""
        gj = GapJunction(conductance=0.1)
        voltages = np.array([-50.0, -70.0])
        adj = np.array([[0, 1], [1, 0]])
        currents = gj.current_matrix(voltages, adj)
        # Neuron 0: g*(V1-V0) = 0.1*(-70-(-50)) = -2.0
        # Neuron 1: g*(V0-V1) = 0.1*(-50-(-70)) = +2.0
        np.testing.assert_allclose(currents, [-2.0, 2.0])

    def test_current_matrix_conserves_current(self) -> None:
        """Currents across the population sum to zero."""
        gj = GapJunction(conductance=0.5)
        voltages = np.array([-50.0, -60.0, -70.0])
        adj = np.ones((3, 3)) - np.eye(3)
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents.sum(), 0.0, atol=1e-10)

    def test_weighted_reciprocal_current_matrix_conserves_current(self) -> None:
        """Weighted reciprocal coupling still conserves current."""
        gj = GapJunction(conductance=0.25)
        voltages = np.array([-50.0, -60.0, -70.0])
        adj = np.array(
            [
                [0.0, 0.5, 0.25],
                [0.5, 0.0, 0.75],
                [0.25, 0.75, 0.0],
            ]
        )
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents.sum(), 0.0, atol=1e-10)

    def test_no_connections_no_current(self) -> None:
        """Without connections no current flows."""
        gj = GapJunction(conductance=1.0)
        voltages = np.array([-50.0, -70.0, -90.0])
        adj = np.zeros((3, 3))
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents, [0.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        ("voltages", "adjacency"),
        [
            (np.array([-50.0, float("nan")]), np.ones((2, 2))),
            (np.array([[-50.0, -60.0]]), np.ones((2, 2))),
            (np.array([-50.0, -60.0]), np.ones((2, 3))),
            (np.array([-50.0, -60.0]), np.array([[0.0, -1.0], [-1.0, 0.0]])),
            (np.array([-50.0, -60.0]), np.array([[0.0, 1.0], [0.0, 0.0]])),
            (np.array([-50.0, -60.0]), np.ones(2)),
            (
                np.array([-50.0, -60.0]),
                np.array([[0.0, float("inf")], [float("inf"), 0.0]]),
            ),
        ],
    )
    def test_invalid_current_matrix_inputs_fail_closed(
        self, voltages: np.ndarray[Any, Any], adjacency: np.ndarray[Any, Any]
    ) -> None:
        """Each invalid voltage or adjacency is rejected."""
        gj = GapJunction(conductance=0.1)
        with pytest.raises(ValueError, match="current_matrix"):
            gj.current_matrix(voltages, adjacency)
