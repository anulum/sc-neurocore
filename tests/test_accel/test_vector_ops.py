"""Tests for vectorized bitstream operations."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream, vec_and, vec_popcount


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


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vector_ops_perf_pack():
    """Benchmark packing a large bitstream."""
    bits = np.random.randint(0, 2, size=100_000, dtype=np.uint8)
    start = time.perf_counter()
    _ = pack_bitstream(bits)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
