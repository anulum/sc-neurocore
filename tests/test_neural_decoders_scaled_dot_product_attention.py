# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScaledDotProductAttention from former test_neural_decoders.py

"""Focused suite: TestScaledDotProductAttention from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403


class TestScaledDotProductAttention:
    def test_identity_keys(self) -> None:
        n, d = 4, 8
        q = np.eye(n, d)
        k = np.eye(n, d)
        v = np.random.default_rng(42).normal(0, 1, (n, d))
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (n, d)

    def test_uniform_attention_on_equal_keys(self) -> None:
        q = np.ones((2, 4))
        k = np.ones((3, 4))
        v = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = scaled_dot_product_attention(q, k, v)
        # All keys equal → uniform weights → output = mean of values
        expected = v.mean(axis=0)
        np.testing.assert_allclose(out[0], expected, atol=1e-10)
