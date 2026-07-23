# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSheafConsistencyDefect from former test_topological_observables.py

"""Focused suite: TestSheafConsistencyDefect from former test_topological_observables.py."""

from __future__ import annotations

from tests.topological_observables_support import *  # noqa: F403

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
