# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVecMux from former test_sc_primitives.py

"""Focused suite: TestVecMux from former test_sc_primitives.py."""

from __future__ import annotations

from tests.sc_primitives_support import *  # noqa: F403


class TestVecMux:
    def test_half_addition(self):
        # MUX with sel=0.5 gives (A + B) / 2
        pa, pb = 0.8, 0.2
        a, length = _bernoulli_packed(pa, 20000, seed=1)
        b, _ = _bernoulli_packed(pb, 20000, seed=2)
        sel, _ = _bernoulli_packed(0.5, 20000, seed=3)
        result = vec_mux(sel, a, b)
        expected = 0.5 * pa + 0.5 * pb  # = 0.5
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)

    def test_sel_one_passes_a(self):
        a, length = _bernoulli_packed(0.7, 5000, seed=10)
        b, _ = _bernoulli_packed(0.3, 5000, seed=20)
        sel = pack_bitstream(np.ones(5000, dtype=np.uint8))
        result = vec_mux(sel, a, b)
        assert np.array_equal(result, a)

    def test_sel_zero_passes_b(self):
        a, length = _bernoulli_packed(0.7, 5000, seed=10)
        b, _ = _bernoulli_packed(0.3, 5000, seed=20)
        sel = pack_bitstream(np.zeros(5000, dtype=np.uint8))
        result = vec_mux(sel, a, b)
        assert np.array_equal(result, b)

    def test_weighted_addition(self):
        # MUX with sel=0.3 gives 0.3*A + 0.7*B
        pa, pb = 0.9, 0.1
        a, length = _bernoulli_packed(pa, 20000, seed=5)
        b, _ = _bernoulli_packed(pb, 20000, seed=6)
        sel, _ = _bernoulli_packed(0.3, 20000, seed=7)
        result = vec_mux(sel, a, b)
        expected = 0.3 * pa + 0.7 * pb  # = 0.34
        np.testing.assert_allclose(_prob(result, length), expected, atol=0.03)
