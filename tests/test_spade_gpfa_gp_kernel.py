# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPKernel from former test_spade_gpfa.py

"""Focused suite: TestGPKernel from former test_spade_gpfa.py."""

from __future__ import annotations

from tests.spade_gpfa_support import *  # noqa: F403


class TestGPKernel:
    def test_shape_and_symmetry(self):
        K = _gp_kernel(50, tau=10.0)
        assert K.shape == (50, 50)
        np.testing.assert_allclose(K, K.T)

    def test_diagonal_equals_sigma_squared(self):
        K = _gp_kernel(30, tau=5.0, sigma=2.0)
        np.testing.assert_allclose(np.diag(K), 4.0)

    def test_positive_definite(self):
        K = _gp_kernel(20, tau=8.0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)
