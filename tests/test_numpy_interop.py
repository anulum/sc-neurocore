# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NumPy zero-copy interop functions

"""Tests for NumPy zero-copy interop functions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

import sc_neurocore_engine as v3


class TestPackBitstreamNumpy:
    """Zero-copy pack_bitstream_numpy tests."""

    def test_basic_pack(self):
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.uint64

    def test_roundtrip(self):
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        recovered = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_large_array(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1_000_000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert packed.dtype == np.uint64
        expected_words = (1_000_000 + 63) // 64
        assert len(packed) == expected_words

    def test_consistency_with_list_variant(self):
        """Numpy and list variants must produce identical results."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1000).astype(np.uint8)
        packed_np = v3.pack_bitstream_numpy(bits)
        packed_list = v3.pack_bitstream(bits.tolist())
        np.testing.assert_array_equal(packed_np, np.array(packed_list, dtype=np.uint64))

    def test_all_zeros(self):
        bits = np.zeros(128, dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert all(w == 0 for w in packed)

    def test_all_ones(self):
        bits = np.ones(64, dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert packed[0] == np.uint64(0xFFFFFFFFFFFFFFFF)


class TestPopcountNumpy:
    """Zero-copy popcount_numpy tests."""

    def test_basic(self):
        packed = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 64

    def test_known_value(self):
        packed = np.array([0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 32

    def test_consistency_with_pack(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_large_array(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1_000_000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_empty(self):
        packed = np.array([], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 0


class TestPopcountRoutesNumpyZeroCopy:
    """``popcount`` takes the zero-copy numpy path yet stays result-identical (KR-4)."""

    def test_numpy_matches_popcount_numpy(self):
        packed = np.array([0xFFFFFFFFFFFFFFFF, 0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
        assert v3.popcount(packed) == v3.popcount_numpy(packed) == 96

    def test_numpy_matches_list_variant(self):
        """The fast numpy path and the list-extract path agree on a large bitstream."""
        rng = np.random.RandomState(7)
        bits = rng.randint(0, 2, 10000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert v3.popcount(packed) == v3.popcount(packed.tolist()) == int(bits.sum())

    def test_numpy_empty(self):
        assert v3.popcount(np.array([], dtype=np.uint64)) == 0
