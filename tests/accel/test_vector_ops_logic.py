# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vector bitstream logic tests

"""Packed AND, XNOR, complement, and multiplexer contracts."""

import numpy as np

from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    unpack_bitstream,
    vec_and,
    vec_mux,
    vec_not,
    vec_xnor,
)


def test_vec_and_basic() -> None:
    """vec_and should compute bitwise AND on packed arrays."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    b = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    out = vec_and(a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 0, 0, 0], dtype=np.uint8))


def test_vec_xnor_matches_equality_per_bit() -> None:
    """vec_xnor sets a bit where the two streams agree (SC bipolar multiply)."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    b = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    out = vec_xnor(a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 0, 0, 1], dtype=np.uint8))


def test_vec_not_complements_each_bit() -> None:
    """vec_not flips every packed bit (SC complement 1 - P(A))."""
    a = pack_bitstream(np.array([1, 0, 1, 0], dtype=np.uint8))
    out = vec_not(a)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([0, 1, 0, 1], dtype=np.uint8))


def test_vec_mux_selects_per_bit() -> None:
    """vec_mux routes A where select is 1 and B where select is 0."""
    select = pack_bitstream(np.array([1, 1, 0, 0], dtype=np.uint8))
    a = pack_bitstream(np.array([1, 1, 1, 1], dtype=np.uint8))
    b = pack_bitstream(np.array([0, 0, 0, 0], dtype=np.uint8))
    out = vec_mux(select, a, b)
    unpacked = unpack_bitstream(out, 4)
    assert np.array_equal(unpacked, np.array([1, 1, 0, 0], dtype=np.uint8))
