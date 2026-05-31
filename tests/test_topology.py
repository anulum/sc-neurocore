# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for topological observables

"""Tests for winding number, Ollivier-Ricci curvature, sheaf defect, connection curvature."""

import numpy as np

from sc_neurocore.math.topology import (
    winding_number,
    ollivier_ricci_curvature,
    sheaf_consistency_defect,
    connection_curvature,
    _lazy_random_walk,
    _minimum_transport_cost,
)


class TestWindingNumber:
    def test_one_full_rotation(self):
        phases = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        assert winding_number(phases) == 1

    def test_two_rotations(self):
        phases = np.linspace(0, 4 * np.pi, 200, endpoint=False)
        assert winding_number(phases) == 2

    def test_no_rotation(self):
        phases = np.ones(50) * 1.5
        assert winding_number(phases) == 0

    def test_negative_rotation(self):
        phases = np.linspace(2 * np.pi, 0, 100, endpoint=False)
        assert winding_number(phases) == -1


class TestOllivierRicciCurvature:
    def test_identical_neighborhoods(self):
        knm = np.ones((4, 4))
        np.fill_diagonal(knm, 0.0)
        kappa = ollivier_ricci_curvature(knm, 0, 1)
        np.testing.assert_allclose(kappa, 2.0 / 3.0, atol=1e-12)

    def test_disconnected_nodes(self):
        knm = np.zeros((4, 4))
        kappa = ollivier_ricci_curvature(knm, 0, 1)
        assert kappa == 0.0

    def test_nearest_neighbor_chain(self):
        knm = np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
        kappa = ollivier_ricci_curvature(knm, 0, 1)
        # Chain: neighborhoods have minimal overlap
        assert kappa < 1.0

    def test_rejects_invalid_coupling_graph(self):
        with np.testing.assert_raises_regex(ValueError, "at least one node"):
            ollivier_ricci_curvature(np.zeros((0, 0)), 0, 0)
        with np.testing.assert_raises_regex(ValueError, "square"):
            ollivier_ricci_curvature(np.ones((2, 3)), 0, 1)
        with np.testing.assert_raises_regex(ValueError, "finite"):
            ollivier_ricci_curvature(np.array([[0.0, np.nan], [1.0, 0.0]]), 0, 1)
        with np.testing.assert_raises_regex(ValueError, "non-negative"):
            ollivier_ricci_curvature(np.array([[0.0, -1.0], [1.0, 0.0]]), 0, 1)
        with np.testing.assert_raises_regex(ValueError, "out of range"):
            ollivier_ricci_curvature(np.eye(2), 0, 2)

    def test_isolated_lazy_random_walk_stays_at_source(self):
        graph = np.zeros((3, 3))

        distribution = _lazy_random_walk(graph, 1)

        np.testing.assert_allclose(distribution, [0.0, 1.0, 0.0])

    def test_transport_helper_handles_empty_or_disconnected_support(self):
        distances = np.array([[0.0, np.inf], [np.inf, 0.0]])

        assert _minimum_transport_cost(np.zeros(2), np.array([1.0, 0.0]), distances) == 0.0
        assert not np.isfinite(
            _minimum_transport_cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]), distances)
        )
        with np.testing.assert_raises_regex(ValueError, "infeasible"):
            _minimum_transport_cost(np.array([1.0, 0.0]), np.array([0.5, 0.0]), np.eye(2))


class TestSheafConsistencyDefect:
    def test_synchronized_is_zero(self):
        phases = np.zeros(5)
        knm = np.ones((5, 5))
        assert sheaf_consistency_defect(phases, knm) == 0.0

    def test_anti_phase_is_positive(self):
        phases = np.array([0.0, np.pi, 0.0, np.pi])
        knm = np.ones((4, 4))
        defect = sheaf_consistency_defect(phases, knm)
        assert defect > 0.0

    def test_zero_coupling_is_zero(self):
        phases = np.random.rand(5) * 2 * np.pi
        knm = np.zeros((5, 5))
        assert sheaf_consistency_defect(phases, knm) == 0.0


class TestConnectionCurvature:
    def test_synchronized_full_coupling(self):
        phases = np.zeros(3)
        knm = np.ones((3, 3))
        F = connection_curvature(phases, knm)
        # cos(0) = 1 everywhere, so F = knm
        np.testing.assert_allclose(F, knm)

    def test_anti_phase_negative(self):
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        F = connection_curvature(phases, knm)
        # cos(pi) = -1, so F[0,1] = -1
        np.testing.assert_allclose(F[0, 1], -1.0)

    def test_diagonal_is_coupling_weighted(self):
        phases = np.array([0.0, 0.0, 0.0])
        knm = np.diag([1.0, 2.0, 3.0])
        F = connection_curvature(phases, knm)
        np.testing.assert_allclose(np.diag(F), [1.0, 2.0, 3.0])
