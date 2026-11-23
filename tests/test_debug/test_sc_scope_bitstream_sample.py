# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamSample from former test_sc_scope.py

"""Focused suite: TestBitstreamSample from former test_sc_scope.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.debug.sc_scope import (
    BitstreamSample,
)
from sc_scope_support import (
    _sample,
)


class TestBitstreamSample:
    """A captured bitstream sample."""

    def test_bit_length(self) -> None:
        """Four 32-bit words are 128 bits."""
        s = _sample(n_words=4)
        assert s.bit_length == 128

    def test_popcount_all_ones(self) -> None:
        """Two all-ones words count 64 ones."""
        words = np.array([0xFFFF_FFFF, 0xFFFF_FFFF], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 64

    def test_popcount_all_zeros(self) -> None:
        """All-zero words count no ones."""
        words = np.array([0, 0], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 0

    def test_density_range(self) -> None:
        """The density lies between 0 and 1."""
        s = _sample(density=0.5)
        assert 0.0 <= s.density <= 1.0

    def test_effective_bits_zero(self) -> None:
        """A constant stream carries no effective bits."""
        words = np.array([0xFFFF_FFFF] * 4, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits == 0.0  # No entropy at p=1

    def test_effective_bits_half(self) -> None:
        """A half-density stream carries effective bits."""
        rng = np.random.default_rng(42)
        words = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits > 0
