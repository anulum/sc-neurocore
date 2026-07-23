# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScDoctor from former test_wave4.py

"""Focused suite: TestScDoctor from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestScDoctor:
    def test_new_defaults(self):
        d = ScDoctor(512, 0.90)
        assert d.current_bitstream_length == 512
        assert d.target_precision == 0.90
        assert not d.error_correction_enabled

    def test_hamming74_roundtrip_all_patterns(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        for data in range(16):
            encoded = d.encode_ecc(data)
            decoded = d.decode_ecc(encoded)
            assert data == decoded, f"Roundtrip failed for {data:#06b}"

    def test_hamming74_single_bit_correction(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        data = 0b1011
        encoded = d.encode_ecc(data)
        for bit in range(7):
            corrupted = encoded ^ (1 << bit)
            recovered = d.decode_ecc(corrupted)
            assert data == recovered, f"Failed to correct bit {bit}"

    def test_ecc_bypass_when_disabled(self):
        d = ScDoctor()
        assert d.encode_ecc(0b1011) == 0b1011
        assert d.decode_ecc(0b1111011) == 0b1011

    def test_adapt_high_correlation_doubles(self):
        d = ScDoctor(256)
        d.adapt(0.20)
        assert d.current_bitstream_length == 512

    def test_adapt_low_correlation_halves(self):
        d = ScDoctor(512)
        d.adapt(0.03)
        assert d.current_bitstream_length == 256

    def test_adapt_floor_at_256(self):
        d = ScDoctor(256)
        d.adapt(0.03)
        assert d.current_bitstream_length == 256

    def test_adapt_enables_ecc_above_2048(self):
        d = ScDoctor(1024)
        d.adapt(0.30)
        assert not d.error_correction_enabled
        d.adapt(0.30)
        assert d.error_correction_enabled

    def test_adapt_disables_ecc_on_halve(self):
        d = ScDoctor(4096)
        d.error_correction_enabled = True
        d.adapt(0.03)
        assert not d.error_correction_enabled
        assert d.current_bitstream_length == 2048

    def test_adapt_mid_correlation_no_change(self):
        d = ScDoctor(512)
        d.adapt(0.10)
        assert d.current_bitstream_length == 512

    def test_encoded_fits_7_bits(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        for data in range(16):
            assert d.encode_ecc(data) < 128

    def test_all_zero_pattern(self):
        d = ScDoctor()
        d.error_correction_enabled = True
        assert d.encode_ecc(0b0000) == 0b0000000

    def test_rust_dispatch_paths(self, monkeypatch: pytest.MonkeyPatch):
        import sc_neurocore.debug.sc_doctor as sc_doctor_mod

        class _FakeRustDoctor:
            @staticmethod
            def py_sc_doctor_adapt(length: int, ecc: bool, corr: float):
                return (length + 16, True)

            @staticmethod
            def py_hamming74_encode(data: int):
                return data ^ 0b1111111

            @staticmethod
            def py_hamming74_decode(encoded: int):
                return encoded ^ 0b1111111

        monkeypatch.setattr(sc_doctor_mod, "_HAS_RUST_DOCTOR", True)
        monkeypatch.setattr(sc_doctor_mod, "_sdc", _FakeRustDoctor())

        d = sc_doctor_mod.ScDoctor(256)
        d.adapt(0.1)
        assert d.current_bitstream_length == 272
        assert d.error_correction_enabled is True
        d.error_correction_enabled = True
        assert d.decode_ecc(d.encode_ecc(0b0110)) == 0b0110
