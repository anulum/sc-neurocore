# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPcaInit from former test_gpfa.py

"""Focused suite: TestPcaInit from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403

class TestPcaInit:
    def test_deterministic_and_shapes(self) -> None:
        Y = np.asarray(_synthetic_trains(6, 200), dtype=np.float64)[:, :30]
        c1, d1, r1, tau1 = gpfa_pca_init(Y, 3, 20.0)
        c2, d2, r2, tau2 = gpfa_pca_init(Y, 3, 20.0)
        npt.assert_array_equal(c1, c2)
        assert c1.shape == (6, 3)
        assert d1.shape == (6,)
        assert r1.shape == (6, 6)
        assert tau1.shape == (3,)
        npt.assert_array_equal(tau1, 40.0)

    def test_sign_convention_max_abs_entry_positive(self) -> None:
        Y = np.asarray(_synthetic_trains(5, 150), dtype=np.float64)[:, :25]
        c, _, _, _ = gpfa_pca_init(Y, 2, 20.0)
        for j in range(c.shape[1]):
            col = c[:, j]
            assert col[np.argmax(np.abs(col))] >= 0.0
