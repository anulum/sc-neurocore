# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vector bitstream packing tests

"""Round-trip, padding, dtype, empty, and input-container packing contracts."""

import numpy as np

from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream


def test_pack_unpack_roundtrip_1d() -> None:
    """Pack/unpack should preserve 1D bitstream."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size)
    assert np.array_equal(bits, unpacked)


def test_pack_unpack_roundtrip_2d() -> None:
    """Pack/unpack should preserve 2D bitstream."""
    bits = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    packed = pack_bitstream(bits)
    # For 2D arrays, unpack returns 2D, use original_shape parameter
    unpacked = unpack_bitstream(packed, bits.size, original_shape=bits.shape)
    assert np.array_equal(bits, unpacked)


def test_pack_bitstream_padding() -> None:
    """Padding should not affect unpacked length."""
    bits = np.random.randint(0, 2, size=70, dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size)
    assert unpacked.size == bits.size


def test_pack_bitstream_dtype() -> None:
    """Packed output should be uint64."""
    bits = np.random.randint(0, 2, size=64, dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert packed.dtype == np.uint64


def test_pack_bitstream_empty() -> None:
    """Empty input should produce empty packed array."""
    bits = np.array([], dtype=np.uint8)
    packed = pack_bitstream(bits)
    assert packed.size == 0


def test_pack_bitstream_accepts_list() -> None:
    """pack_bitstream should accept Python lists."""
    bits = [1, 0, 1, 0, 1]
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, len(bits))
    assert np.array_equal(np.array(bits, dtype=np.uint8), unpacked)


def test_unpack_2d_without_shape_splits_evenly_per_batch() -> None:
    """A 2D unpack with no original_shape recovers original_length // batch bits per row."""
    bits = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    packed = pack_bitstream(bits)
    unpacked = unpack_bitstream(packed, bits.size, original_shape=None)
    assert np.array_equal(unpacked, bits)
