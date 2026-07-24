# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamSample from former test_sc_scope.py

"""Focused suite: TestBitstreamSample from former test_sc_scope.py."""

from __future__ import annotations

from sc_scope_support import *  # noqa: F403


class TestBitstreamSample:
    def test_bit_length(self):
        s = _sample(n_words=4)
        assert s.bit_length == 128

    def test_popcount_all_ones(self):
        words = np.array([0xFFFF_FFFF, 0xFFFF_FFFF], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 64

    def test_popcount_all_zeros(self):
        words = np.array([0, 0], dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.popcount == 0

    def test_density_range(self):
        s = _sample(density=0.5)
        assert 0.0 <= s.density <= 1.0

    def test_effective_bits_zero(self):
        words = np.array([0xFFFF_FFFF] * 4, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits == 0.0  # No entropy at p=1

    def test_effective_bits_half(self):
        rng = np.random.default_rng(42)
        words = rng.integers(0, 0xFFFF_FFFF, size=16, dtype=np.uint32)
        s = BitstreamSample(0, 0, 0, words)
        assert s.effective_bits > 0
