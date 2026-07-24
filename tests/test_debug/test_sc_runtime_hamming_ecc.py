# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHammingECC from former test_sc_runtime.py

"""Focused suite: TestHammingECC from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403


class TestHammingECC:
    def test_roundtrip_all_patterns(self):
        ecc = HammingECC()
        for data in range(16):
            encoded = ecc.encode(data)
            decoded = ecc.decode(encoded)
            assert decoded == data, f"Roundtrip failed for {data}"

    def test_single_bit_correction(self):
        ecc = HammingECC()
        data = 0b1011
        encoded = ecc.encode(data)
        for bit in range(7):
            corrupted = encoded ^ (1 << bit)
            recovered = ecc.decode(corrupted)
            assert recovered == data, f"Failed to correct bit {bit}"

    def test_encoded_fits_7_bits(self):
        ecc = HammingECC()
        for data in range(16):
            assert ecc.encode(data) < 128

    def test_bitstream_roundtrip(self):
        ecc = HammingECC()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)

    def test_bitstream_ecc_detects_corruption(self):
        ecc = HammingECC()
        bs = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[3] ^= 1
        decoded = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
