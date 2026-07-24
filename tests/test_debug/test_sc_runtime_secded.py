# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSECDED from former test_sc_runtime.py

"""Focused suite: TestSECDED from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403


class TestSECDED:
    def test_roundtrip_all_patterns(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            encoded = ecc.encode(data)
            decoded, uncorrectable = ecc.decode(encoded)
            assert decoded == data
            assert not uncorrectable

    def test_encoded_fits_8_bits(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            assert ecc.encode(data) < 256

    def test_single_bit_correction(self):
        ecc = SECDEC_ECC()
        for data in range(16):
            encoded = ecc.encode(data)
            for bit in range(8):
                corrupted = encoded ^ (1 << bit)
                decoded, uncorrectable = ecc.decode(corrupted)
                assert decoded == data, f"Failed 1-bit correction for data={data}, bit={bit}"
                assert not uncorrectable

    def test_double_bit_detection(self):
        ecc = SECDEC_ECC()
        data = 0b1010
        encoded = ecc.encode(data)
        # Flip two bits
        corrupted = encoded ^ 0b11
        _, uncorrectable = ecc.decode(corrupted)
        assert uncorrectable

    def test_bitstream_roundtrip(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        decoded, n_unc = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
        assert n_unc == 0

    def test_bitstream_single_bit_correction(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[5] ^= 1  # corrupt 1 bit
        decoded, n_unc = ecc.decode_bitstream(encoded)
        np.testing.assert_array_equal(decoded[: len(bs)], bs)
        assert n_unc == 0

    def test_bitstream_double_bit_detected(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 0], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        encoded[0] ^= 1
        encoded[1] ^= 1
        _, n_unc = ecc.decode_bitstream(encoded)
        assert n_unc > 0

    def test_secded_8_bit_encoding(self):
        ecc = SECDEC_ECC()
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        encoded = ecc.encode_bitstream(bs)
        assert len(encoded) == 8  # 4 data → 8 SECDED
