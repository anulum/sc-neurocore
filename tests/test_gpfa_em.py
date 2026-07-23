# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEm from former test_gpfa.py

"""Focused suite: TestEm from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403

class TestEm:
    def test_converges_and_is_deterministic(self) -> None:
        Y = np.asarray(_synthetic_trains(8, 600), dtype=np.float64)[:, :30]
        c0, d0, r0, tau = gpfa_pca_init(Y, 3, 20.0)
        x1, c1, _, _, ll1 = gpfa_em(Y, c0, d0, r0, tau, 40, 1e-4)
        x2, c2, _, _, ll2 = gpfa_em(Y, c0, d0, r0, tau, 40, 1e-4)
        npt.assert_array_equal(x1, x2)
        npt.assert_array_equal(c1, c2)
        assert ll1[-1] >= ll1[0]
        assert len(ll1) < 40  # converged before the cap
        assert ll1 == ll2

    def test_respects_max_iter_without_convergence(self) -> None:
        Y = np.asarray(_synthetic_trains(6, 400), dtype=np.float64)[:, :25]
        c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
        _, _, _, _, ll = gpfa_em(Y, c0, d0, r0, tau, 3, 1e-12)
        assert len(ll) == 3
