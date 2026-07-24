# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGpKernel from former test_gpfa.py

"""Focused suite: TestGpKernel from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


class TestGpKernel:
    def test_shape_diagonal_and_symmetry(self) -> None:
        k = _gp_kernel(10, 5.0, 1.0)
        assert k.shape == (10, 10)
        npt.assert_allclose(np.diag(k), 1.0)
        npt.assert_allclose(k, k.T)

    def test_decays_with_distance(self) -> None:
        k = _gp_kernel(20, 3.0)
        assert k[0, 1] > k[0, 10]
