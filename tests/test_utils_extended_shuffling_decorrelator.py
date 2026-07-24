# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestShufflingDecorrelator from former test_utils_extended.py

"""Focused suite: TestShufflingDecorrelator from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403


class TestShufflingDecorrelator:
    def test_preserves_bit_count(self):
        """Shuffling must keep exact same number of ones."""
        bs = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
        dec = ShufflingDecorrelator(window_size=8, seed=42)
        result = dec.process(bs)
        assert result.sum() == bs.sum()
        assert len(result) == len(bs)

    def test_preserves_probability(self):
        """On a long bitstream, probability should be unchanged."""
        rng = np.random.default_rng(7)
        bs = (rng.random(1024) < 0.3).astype(np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=99)
        result = dec.process(bs)
        assert result.mean() == pytest.approx(bs.mean(), abs=1e-12)

    def test_output_length_preserved_non_divisible(self):
        """Length not divisible by window_size should still return correct length."""
        bs = np.ones(100, dtype=np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=1)
        result = dec.process(bs)
        assert len(result) == 100

    def test_changes_bit_positions(self):
        """Shuffled output should differ from input in at least some positions."""
        rng = np.random.default_rng(0)
        bs = (rng.random(256) < 0.5).astype(np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=42)
        result = dec.process(bs)
        # With p=0.5 and window=16, very unlikely to be identical
        assert not np.array_equal(result, bs)
