# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC division (CORDIV)

"""Tests for stochastic computing division circuit."""

import numpy as np
import pytest

from sc_neurocore.utils.bitstreams import (
    sc_divide,
    bitstream_to_probability,
)


class TestSCDivide:
    def test_half_divided_by_one(self):
        """P(x)=0.5 / P(y)=1.0 ≈ 0.5."""
        rng = np.random.RandomState(42)
        length = 10000
        x = (rng.random(length) < 0.5).astype(np.uint8)
        y = np.ones(length, dtype=np.uint8)
        z = sc_divide(x, y)
        p_z = bitstream_to_probability(z)
        np.testing.assert_allclose(p_z, 0.5, atol=0.05)

    def test_quarter_divided_by_half(self):
        """P(x)=0.25 / P(y)=0.5 ≈ 0.5."""
        rng = np.random.RandomState(42)
        length = 10000
        x = (rng.random(length) < 0.25).astype(np.uint8)
        y = (rng.random(length) < 0.5).astype(np.uint8)
        z = sc_divide(x, y)
        p_z = bitstream_to_probability(z)
        np.testing.assert_allclose(p_z, 0.5, atol=0.1)

    def test_identity_division(self):
        """P(x)/P(x) ≈ 1.0 (x divides itself)."""
        rng = np.random.RandomState(42)
        length = 10000
        x = (rng.random(length) < 0.6).astype(np.uint8)
        z = sc_divide(x, x)
        p_z = bitstream_to_probability(z)
        # x/x should be close to 1 when x is not 0
        np.testing.assert_allclose(p_z, 1.0, atol=0.05)

    def test_zero_numerator(self):
        """P(x)=0 / P(y)=0.5 → 0."""
        length = 1000
        x = np.zeros(length, dtype=np.uint8)
        rng = np.random.RandomState(42)
        y = (rng.random(length) < 0.5).astype(np.uint8)
        z = sc_divide(x, y)
        p_z = bitstream_to_probability(z)
        assert p_z < 0.05

    def test_output_is_binary(self):
        """Output should be {0, 1} only."""
        rng = np.random.RandomState(42)
        length = 100
        x = (rng.random(length) < 0.3).astype(np.uint8)
        y = (rng.random(length) < 0.7).astype(np.uint8)
        z = sc_divide(x, y)
        assert set(np.unique(z)).issubset({0, 1})

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            sc_divide(np.array([0, 1], dtype=np.uint8), np.array([0, 1, 0], dtype=np.uint8))

    def test_small_ratio(self):
        """P(x)=0.1 / P(y)=0.5 ≈ 0.2."""
        rng = np.random.RandomState(42)
        length = 10000
        x = (rng.random(length) < 0.1).astype(np.uint8)
        y = (rng.random(length) < 0.5).astype(np.uint8)
        z = sc_divide(x, y)
        p_z = bitstream_to_probability(z)
        np.testing.assert_allclose(p_z, 0.2, atol=0.1)
