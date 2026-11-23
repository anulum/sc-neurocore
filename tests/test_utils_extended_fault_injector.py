# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjector from former test_utils_extended.py

"""Focused suite: TestFaultInjector from former test_utils_extended.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.utils.fault_injection import (
    FaultInjector,
)


class TestFaultInjector:
    """Injecting faults into bitstreams."""

    def test_bit_flip_zero_rate(self) -> None:
        """Error rate 0 should return unchanged bitstream."""
        bs = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.0)
        np.testing.assert_array_equal(result, bs)

    def test_bit_flip_full_rate(self) -> None:
        """Error rate 1.0 should flip every bit."""
        bs = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        np.random.seed(0)
        result = FaultInjector.inject_bit_flips(bs, 1.0)
        expected = 1 - bs
        np.testing.assert_array_equal(result, expected)

    def test_bit_flip_output_binary(self) -> None:
        """Output must be strictly 0/1."""
        np.random.seed(0)
        bs = np.zeros(1000, dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.5)
        assert set(np.unique(result)).issubset({0, 1})

    def test_bit_flip_approximate_rate(self) -> None:
        """Fraction of flipped bits should roughly match error_rate."""
        np.random.seed(42)
        bs = np.zeros(10000, dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.1)
        flip_rate = result.mean()  # started all zeros, flips become ones
        assert flip_rate == pytest.approx(0.1, abs=0.02)

    def test_stuck_at_zero(self) -> None:
        """Stuck-at-0 forces selected bits to 0."""
        np.random.seed(0)
        bs = np.ones(1000, dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.3, value=0)
        # Some bits should now be 0
        assert result.sum() < 1000
        # Rate should be approximately 30% zeros
        zero_frac = 1.0 - result.mean()
        assert zero_frac == pytest.approx(0.3, abs=0.05)

    def test_stuck_at_one(self) -> None:
        """Stuck-at-1 forces selected bits to 1."""
        np.random.seed(0)
        bs = np.zeros(1000, dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.3, value=1)
        one_frac = result.mean()
        assert one_frac == pytest.approx(0.3, abs=0.05)

    def test_stuck_at_preserves_shape(self) -> None:
        """Stuck-at faults keep the array shape."""
        bs = np.zeros((4, 5), dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.5, value=1)
        assert result.shape == (4, 5)
