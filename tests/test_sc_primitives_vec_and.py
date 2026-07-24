# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVecAnd from former test_sc_primitives.py

"""Focused suite: TestVecAnd from former test_sc_primitives.py."""

from __future__ import annotations

from tests.sc_primitives_support import *  # noqa: F403


class TestVecAnd:
    def test_multiply(self):
        pa, pb = 0.6, 0.4
        a, length = _bernoulli_packed(pa, 10000, seed=100)
        b, _ = _bernoulli_packed(pb, 10000, seed=200)
        result = vec_and(a, b)
        expected = pa * pb  # = 0.24
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)
