# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVecXnor from former test_sc_primitives.py

"""Focused suite: TestVecXnor from former test_sc_primitives.py."""

from __future__ import annotations

from tests.sc_primitives_support import *  # noqa: F403


class TestVecXnor:
    def test_bipolar_multiply(self):
        # XNOR on unipolar streams gives P(A)*P(B) + (1-P(A))*(1-P(B))
        pa, pb = 0.8, 0.6
        a, length = _bernoulli_packed(pa, 10000, seed=10)
        b, _ = _bernoulli_packed(pb, 10000, seed=20)
        result = vec_xnor(a, b)
        expected = pa * pb + (1 - pa) * (1 - pb)
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)

    def test_self_xnor_is_all_ones(self):
        packed, length = _bernoulli_packed(0.5, 1000, seed=42)
        result = vec_xnor(packed, packed)
        # XNOR(x, x) = NOT(XOR(x, x)) = NOT(0) = all 1s
        assert _prob(result, length) > 0.99
