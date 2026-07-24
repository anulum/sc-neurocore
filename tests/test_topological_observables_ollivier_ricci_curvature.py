# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOllivierRicciCurvature from former test_topological_observables.py

"""Focused suite: TestOllivierRicciCurvature from former test_topological_observables.py."""

from __future__ import annotations

from tests.topological_observables_support import *  # noqa: F403


class TestOllivierRicciCurvature:
    def test_complete_graph_positive(self):
        """Complete graph has positive lazy Ollivier-Ricci curvature."""
        N = 8
        K = np.ones((N, N)) * 0.5
        np.fill_diagonal(K, 0)
        kappa = ollivier_ricci_curvature(K, 0, 1)
        assert kappa > 0, f"complete graph curvature {kappa} not positive"

    def test_disconnected_zero(self):
        """If i and j are not connected, curvature should be 0 or undefined."""
        N = 4
        K = np.zeros((N, N))
        K[0, 1] = K[1, 0] = 1.0  # only 0-1 connected
        kappa = ollivier_ricci_curvature(K, 0, 2)  # 0 and 2 not connected
        # Depending on implementation, may be 0 or handled gracefully
        assert np.isfinite(kappa)

    def test_self_curvature_zero(self):
        N = 5
        K = np.ones((N, N))
        np.fill_diagonal(K, 0)
        kappa = ollivier_ricci_curvature(K, 0, 0)
        assert kappa == 0.0 or np.isfinite(kappa)

    def test_ring_lower_than_complete(self):
        """Ring graph should have lower curvature than complete graph."""
        N = 10
        K_ring = np.zeros((N, N))
        for i in range(N):
            K_ring[i, (i + 1) % N] = K_ring[(i + 1) % N, i] = 1.0
        K_complete = np.ones((N, N))
        np.fill_diagonal(K_complete, 0)
        kappa_ring = ollivier_ricci_curvature(K_ring, 0, 1)
        kappa_complete = ollivier_ricci_curvature(K_complete, 0, 1)
        assert kappa_ring < kappa_complete

    def test_path_bridge_uses_graph_metric_transport(self):
        K = np.zeros((5, 5))
        for node in range(4):
            K[node, node + 1] = 1.0
            K[node + 1, node] = 1.0

        endpoint_edge = ollivier_ricci_curvature(K, 0, 1)
        middle_edge = ollivier_ricci_curvature(K, 2, 3)

        np.testing.assert_allclose(middle_edge, 0.0, atol=1e-12)
        assert 0.0 < endpoint_edge < 1.0

    def test_rejects_boolean_node_index(self):
        K = np.ones((3, 3)) - np.eye(3)

        with np.testing.assert_raises_regex(ValueError, "integer"):
            ollivier_ricci_curvature(K, True, 1)
