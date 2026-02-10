"""Equivalence: v2 pack/unpack/popcount vs v3."""

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    unpack_bitstream as v2_unpack,
    vec_popcount as v2_popcount,
)
import sc_neurocore_engine as v3


class TestPackEquivalence:
    @pytest.mark.parametrize("length", [64, 128, 256, 1024, 1025, 4096])
    def test_pack_1d(self, length):
        bits = np.random.RandomState(42).randint(0, 2, length).astype(np.uint8)
        v2_result = v2_pack(bits)
        v3_result = np.asarray(v3.pack_bitstream(bits), dtype=np.uint64)
        np.testing.assert_array_equal(v2_result, v3_result)

    def test_unpack_roundtrip(self, sample_bitstream):
        packed = v2_pack(sample_bitstream)
        v2_unpacked = v2_unpack(packed, original_length=sample_bitstream.size)
        v3_unpacked = np.asarray(
            v3.unpack_bitstream(packed, original_length=sample_bitstream.size),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(v2_unpacked, v3_unpacked)

    def test_popcount(self, sample_bitstream):
        packed = v2_pack(sample_bitstream)
        v2_count = int(v2_popcount(packed))
        v3_count = int(v3.popcount(packed))
        assert v2_count == v3_count
