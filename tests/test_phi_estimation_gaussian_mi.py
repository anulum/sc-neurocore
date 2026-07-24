# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGaussianMI from former test_phi_estimation.py

"""Focused suite: TestGaussianMI from former test_phi_estimation.py."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


class TestGaussianMI:
    def test_nonnegative_and_symmetric(self) -> None:
        rng = np.random.RandomState(1)
        x = rng.randn(2, 120)
        y = rng.randn(2, 120)
        mi = _gaussian_mi(x, y)
        assert mi >= 0.0
        npt.assert_allclose(mi, _gaussian_mi(y, x), atol=1e-12)

    def test_single_channel_blocks(self) -> None:
        # 1-row blocks must use the unbiased variance (ddof=1) without error.
        rng = np.random.RandomState(2)
        mi = _gaussian_mi(rng.randn(1, 80), rng.randn(1, 80))
        assert mi >= 0.0
