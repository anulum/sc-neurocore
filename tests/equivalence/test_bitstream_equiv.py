# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equivalence: v2 pack/unpack/popcount vs v3

"""Equivalence: v2 pack/unpack/popcount vs v3."""

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    unpack_bitstream as v2_unpack,
    vec_popcount as v2_popcount,
)
from tests.engine_requirement import require_engine

require_engine()

import sc_neurocore_engine as v3


class TestPackEquivalence:
    """Bit packing agrees between the v2 Python and v3 engine implementations."""

    @pytest.mark.parametrize("length", [64, 128, 256, 1024, 1025, 4096])
    def test_pack_1d(self, length: int) -> None:
        """The v2 and v3 implementations pack a one-dimensional bitstream identically."""
        bits = np.random.RandomState(42).randint(0, 2, length).astype(np.uint8)
        v2_result = v2_pack(bits)
        v3_result = np.asarray(v3.pack_bitstream(bits), dtype=np.uint64)
        np.testing.assert_array_equal(v2_result, v3_result)

    def test_unpack_roundtrip(self, sample_bitstream: np.ndarray) -> None:
        """Packing then unpacking returns the original bits in both implementations."""
        packed = v2_pack(sample_bitstream)
        v2_unpacked = v2_unpack(packed, original_length=sample_bitstream.size)
        raw = v3.unpack_bitstream(packed, original_length=sample_bitstream.size)
        v3_unpacked = (
            np.frombuffer(raw, dtype=np.uint8)
            if isinstance(raw, bytes)
            else np.asarray(raw, dtype=np.uint8)
        )
        np.testing.assert_array_equal(v2_unpacked, v3_unpacked)

    def test_popcount(self, sample_bitstream: np.ndarray) -> None:
        """Both implementations count the same set bits."""
        packed = v2_pack(sample_bitstream)
        v2_count = int(v2_popcount(packed))
        v3_count = int(v3.popcount(packed))
        assert v2_count == v3_count
