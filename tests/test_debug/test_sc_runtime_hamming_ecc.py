# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHammingECC from former test_sc_runtime.py

"""Focused suite: TestHammingECC from former test_sc_runtime.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.control.sc_runtime import (
    HammingECC,
)


class TestHammingECC:
    """Hamming(7,4) error correction."""

    def test_roundtrip_all_patterns(self) -> None:
        """Every 4-bit pattern decodes back unchanged."""
        ecc = HammingECC()
        for data in range(16):
            encoded = ecc.encode(data)
            decoded = ecc.decode(encoded)
            assert decoded == data, f"Roundtrip failed for {data}"

    def test_single_bit_correction(self) -> None:
        """Any single flipped bit is corrected."""
        ecc = HammingECC()
        data = 0b1011
        encoded = ecc.encode(data)
        for bit in range(7):
            corrupted = encoded ^ (1 << bit)
            recovered = ecc.decode(corrupted)
            assert recovered == data, f"Failed to correct bit {bit}"

    def test_encoded_fits_7_bits(self) -> None:
        """Every codeword fits in 7 bits."""
        ecc = HammingECC()
        for data in range(16):
            assert ecc.encode(data) < 128

    def test_bitstream_roundtrip(self) -> None:
        """A protected bitstream decodes back unchanged."""
        ecc = HammingECC()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)

    def test_bitstream_ecc_detects_corruption(self) -> None:
        """A corrupted protected bitstream decodes back unchanged."""
        ecc = HammingECC()
        bs = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[3] ^= 1
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
