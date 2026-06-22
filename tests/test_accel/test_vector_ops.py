# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for vectorized bitstream operations

"""Tests for vectorized bitstream operations."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    unpack_bitstream,
    vec_and,
    vec_mux,
    vec_not,
    vec_popcount,
    vec_xnor,
)


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_pack_unpack_roundtrip_1d():
    """Pack/unpack should preserve 1D bitstream."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size)
    assert np.array_equal(bits, unpacked)


def test_pack_unpack_roundtrip_2d():
    """Pack/unpack should preserve 2D bitstream."""
    bits = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    packed = pack_bitstream(bits)
    # For 2D arrays, unpack returns 2D, use original_shape parameter
    unpacked = unpack_bitstream(packed, bits.size, original_shape=bits.shape)
    assert np.array_equal(bits, unpacked)


def test_pack_bitstream_padding():
    """Padding should not affect unpacked length."""
    bits = np.random.randint(0, 2, size=70, dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size)
    assert unpacked.size == bits.size


def test_pack_bitstream_dtype():
    """Packed output should be uint64."""
    bits = np.random.randint(0, 2, size=64, dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert packed.dtype == np.uint64


def test_pack_bitstream_empty():
    """Empty input should produce empty packed array."""
    bits = np.array([], dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert packed.size == 0


def test_vec_and_basic():
    """vec_and should compute bitwise AND on packed arrays."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    b = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    out = vec_and(a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 0, 0, 0], dtype=np.uint8))


def test_vec_popcount_known():
    """vec_popcount should count total set bits."""
    bits = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    packed = pack_bitstream(bits)
    count = vec_popcount(packed)
    assert count == 4


def test_vec_popcount_zero():
    """Popcount of all-zero input should be zero."""
    bits = np.zeros(128, dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert vec_popcount(packed) == 0


def test_pack_bitstream_accepts_list():
    """pack_bitstream should accept Python lists."""
    bits = [1, 0, 1, 0, 1]
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, len(bits))
    assert np.array_equal(np.array(bits, dtype=np.uint8), unpacked)


def test_vec_xnor_matches_equality_per_bit():
    """vec_xnor sets a bit where the two streams agree (SC bipolar multiply)."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    b = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    out = vec_xnor(a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 0, 0, 1], dtype=np.uint8))


def test_vec_not_complements_each_bit():
    """vec_not flips every packed bit (SC complement 1 - P(A))."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    out = vec_not(a)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([0, 1, 0, 1], dtype=np.uint8))


def test_vec_mux_selects_per_bit():
    """vec_mux routes A where select is 1 and B where select is 0."""
    select = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    a = pack_bitstream(np.array([1, 1, 1, 1], dtype=np.uint8))
    b = pack_bitstream(np.array([0, 0, 0, 0], dtype=np.uint8))
    out = vec_mux(select, a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 1, 0, 0], dtype=np.uint8))


def test_unpack_2d_without_shape_splits_evenly_per_batch():
    """A 2D unpack with no original_shape recovers original_length // batch bits per row."""
    bits = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size, original_shape=None)
    assert np.array_equal(unpacked, bits)


def test_pack_bitstream_rejects_3d_input():
    """pack_bitstream accepts only 1D or 2D arrays."""
    with pytest.raises(ValueError, match="Expected 1D or 2D array"):
        pack_bitstream(np.zeros((2, 2, 2), dtype=np.uint8))


def test_unpack_bitstream_rejects_3d_packed():
    """unpack_bitstream accepts only 1D or 2D packed arrays."""
    with pytest.raises(ValueError, match="Expected 1D or 2D packed array"):
        unpack_bitstream(np.zeros((2, 2, 2), dtype=np.uint64), 8)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vector_ops_perf_pack():
    """Benchmark packing a large bitstream."""
    bits = np.random.randint(0, 2, size=100_000, dtype=np.uint8)
    start = time.perf_counter()
    _ = pack_bitstream(bits)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
