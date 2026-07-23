# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConnectionCurvature from former test_topological_observables.py

"""Focused suite: TestConnectionCurvature from former test_topological_observables.py."""

from __future__ import annotations

from tests.topological_observables_support import *  # noqa: F403

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
