# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarMultiply from former test_bipolar_sc.py

"""Focused suite: TestBipolarMultiply from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403


class TestBipolarMultiply:
    def test_xnor_same_inputs(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        result = bipolar_multiply(a, a)
        assert (result == 1).all()

    def test_xnor_opposite_inputs(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        b = 1 - a
        result = bipolar_multiply(a, b)
        assert (result == 0).all()

    def test_xnor_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            bipolar_multiply(np.array([1, 0], dtype=np.uint8), np.array([1], dtype=np.uint8))

    def test_statistical_multiplication(self):
        rng = np.random.default_rng(42)
        L = 100000
        for va, vb in [(0.5, 0.5), (-0.5, 0.5), (0.8, -0.3)]:
            a = bipolar_encode(va, L, rng=rng)
            b = bipolar_encode(vb, L, rng=rng)
            product = bipolar_multiply(a, b)
            decoded = bipolar_decode(product)
            expected = va * vb
            assert abs(decoded - expected) < 0.03, f"{va}*{vb}: expected={expected}, got={decoded}"
