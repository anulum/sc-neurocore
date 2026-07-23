# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTernaryWeights from former test_qat.py

"""Focused suite: TestTernaryWeights from former test_qat.py."""

from __future__ import annotations

from tests.qat_support import *  # noqa: F403

class TestTernaryWeights:
    def test_ternary_values(self):
        tw = TernaryWeights()
        t = tw.quantize(np.random.randn(10, 10))
        assert set(np.unique(t)).issubset({-1.0, 0.0, 1.0})

    def test_sparsity(self):
        tw = TernaryWeights(threshold_ratio=0.5)
        s = tw.sparsity(np.random.randn(100, 100))
        assert 0 < s < 1

    def test_higher_threshold_more_sparse(self):
        w = np.random.randn(100, 100)
        low = TernaryWeights(threshold_ratio=0.3).sparsity(w)
        high = TernaryWeights(threshold_ratio=0.9).sparsity(w)
        assert high > low

    def test_all_zero_input(self):
        tw = TernaryWeights()
        t = tw.quantize(np.zeros((5, 5)))
        np.testing.assert_array_equal(t, 0.0)
