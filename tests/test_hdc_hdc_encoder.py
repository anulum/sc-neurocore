# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHDCEncoder from former test_hdc.py

"""Focused suite: TestHDCEncoder from former test_hdc.py."""

from __future__ import annotations

from tests.hdc_support import *  # noqa: F403

class TestHDCEncoder:
    def test_random_vector_shape(self):
        enc = HDCEncoder(dim=1000)
        v = enc.generate_random_vector()
        assert v.shape == (1000,)
        assert v.dtype == np.uint8
        assert set(np.unique(v)).issubset({0, 1})

    def test_random_vector_approximately_balanced(self):
        np.random.seed(42)
        enc = HDCEncoder(dim=10000)
        v = enc.generate_random_vector()
        assert 0.45 < v.mean() < 0.55

    def test_bind_xor(self):
        enc = HDCEncoder(dim=100)
        a = np.array([1, 0, 1, 0, 1] * 20, dtype=np.uint8)
        b = np.array([1, 1, 0, 0, 1] * 20, dtype=np.uint8)
        c = enc.bind(a, b)
        expected = np.bitwise_xor(a, b)
        np.testing.assert_array_equal(c, expected)

    def test_bind_self_inverse(self):
        np.random.seed(0)
        enc = HDCEncoder(dim=1000)
        a = enc.generate_random_vector()
        b = enc.generate_random_vector()
        bound = enc.bind(a, b)
        recovered = enc.bind(bound, b)
        np.testing.assert_array_equal(recovered, a)

    def test_bundle_majority(self):
        enc = HDCEncoder(dim=10)
        v1 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
        v2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        v3 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
        result = enc.bundle([v1, v2, v3])
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})

    def test_bundle_empty(self):
        enc = HDCEncoder(dim=10)
        result = enc.bundle([])
        np.testing.assert_array_equal(result, 0)

    def test_permute(self):
        enc = HDCEncoder(dim=5)
        v = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        p = enc.permute(v, shifts=2)
        # np.roll with positive shift = right circular shift
        np.testing.assert_array_equal(p, [0, 0, 1, 0, 0])
