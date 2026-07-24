# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOllivierRicciCurvature from former test_topology.py

"""Focused suite: TestOllivierRicciCurvature from former test_topology.py."""

from __future__ import annotations

from tests.topology_support import *  # noqa: F403


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
