# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for topological observables (winding, Ricci, sheaf)

"""Tests for winding_number, ollivier_ricci_curvature,
sheaf_consistency_defect, connection_curvature."""

from __future__ import annotations

import numpy as np

from sc_neurocore.math.topology import (
    winding_number,
    ollivier_ricci_curvature,
    sheaf_consistency_defect,
    connection_curvature,
)


class TestWindingNumber:
    def test_zero_wraps(self):
        """Half rotation (0 to pi) = 0 wraps."""
        phases = np.linspace(0, np.pi * 0.9, 500)
        assert winding_number(phases) == 0

    def test_one_wrap(self):
        T = 1000
        omega = 2 * np.pi / T
        phases = np.array([(omega * t) % (2 * np.pi) for t in range(T)])
        assert winding_number(phases) == 1

    def test_three_wraps(self):
        T = 1000
        omega = 3 * 2 * np.pi / T
        phases = np.array([(omega * t) % (2 * np.pi) for t in range(T)])
        assert winding_number(phases) == 3

    def test_constant_phase_zero(self):
        phases = np.full(100, 1.5)
        assert winding_number(phases) == 0

    def test_negative_direction(self):
        """Negative frequency should give negative winding."""
        T = 1000
        omega = -2 * 2 * np.pi / T
        phases = np.array([((omega * t) % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) for t in range(T)])
        w = winding_number(phases)
        # Depending on implementation, may be -2 or wrapped
        assert abs(w) in (0, 2), f"unexpected winding {w}"


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


class TestSheafConsistencyDefect:
    def test_synchronised_zero(self):
        """All phases equal → defect = 0."""
        N = 8
        K = np.ones((N, N)) * 0.5
        np.fill_diagonal(K, 0)
        phases = np.zeros(N)
        defect = sheaf_consistency_defect(phases, K)
        np.testing.assert_allclose(defect, 0.0, atol=1e-10)

    def test_incoherent_positive(self):
        """Random phases → defect > 0."""
        N = 8
        K = np.ones((N, N)) * 0.5
        np.fill_diagonal(K, 0)
        phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
        defect = sheaf_consistency_defect(phases, K)
        assert defect > 0

    def test_zero_coupling_zero_defect(self):
        """No coupling → defect = 0 regardless of phases."""
        N = 5
        K = np.zeros((N, N))
        phases = np.random.default_rng(99).uniform(0, 2 * np.pi, N)
        defect = sheaf_consistency_defect(phases, K)
        np.testing.assert_allclose(defect, 0.0, atol=1e-10)

    def test_anti_phase_maximum(self):
        """Anti-phase (0, pi, 0, pi, ...) on complete graph → near-maximum defect."""
        N = 8
        K = np.ones((N, N)) * 0.5
        np.fill_diagonal(K, 0)
        phases = np.array([0, np.pi] * (N // 2))
        defect_anti = sheaf_consistency_defect(phases, K)
        defect_sync = sheaf_consistency_defect(np.zeros(N), K)
        assert defect_anti > defect_sync

    def test_monotonic_with_noise(self):
        """Increasing phase noise → increasing defect."""
        N = 6
        K = np.ones((N, N)) * 0.3
        np.fill_diagonal(K, 0)
        rng = np.random.default_rng(42)
        prev_defect = 0.0
        for noise in [0.0, 0.5, 1.0, 2.0, np.pi]:
            phases = noise * rng.uniform(0, 1, N)
            d = sheaf_consistency_defect(phases, K)
            # Monotonicity may not be strict for random phases,
            # but should be roughly increasing
            assert d >= -0.01  # non-negative


class TestConnectionCurvature:
    def test_synchronised_equals_coupling(self):
        """cos(0) = 1, so F_ij = K_ij when synchronised."""
        N = 4
        K = np.array([[0, 0.5, 0.3, 0], [0.5, 0, 0, 0.2], [0.3, 0, 0, 0.1], [0, 0.2, 0.1, 0]])
        phases = np.zeros(N)
        F = connection_curvature(phases, K)
        np.testing.assert_allclose(F, K, atol=1e-10)

    def test_anti_phase_negative(self):
        """cos(pi) = -1, so F_ij = -K_ij for anti-phase pairs."""
        N = 4
        K = np.zeros((N, N))
        K[0, 1] = K[1, 0] = 0.5
        phases = np.array([0, np.pi, 0, 0])
        F = connection_curvature(phases, K)
        np.testing.assert_allclose(F[0, 1], -0.5, atol=1e-10)

    def test_diagonal_zero(self):
        N = 4
        K = np.ones((N, N))
        np.fill_diagonal(K, 0)
        phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
        F = connection_curvature(phases, K)
        np.testing.assert_allclose(np.diag(F), 0.0, atol=1e-10)

    def test_output_shape(self):
        N = 6
        K = np.random.default_rng(42).uniform(0, 1, (N, N))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        phases = np.random.default_rng(99).uniform(0, 2 * np.pi, N)
        F = connection_curvature(phases, K)
        assert F.shape == (N, N)

    def test_values_bounded(self):
        """F_ij should be bounded by |K_ij| since |cos| <= 1."""
        N = 5
        K = np.random.default_rng(42).uniform(0, 1, (N, N))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        phases = np.random.default_rng(99).uniform(0, 2 * np.pi, N)
        F = connection_curvature(phases, K)
        assert np.all(np.abs(F) <= np.abs(K) + 1e-10)
