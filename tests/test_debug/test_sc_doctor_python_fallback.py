# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the ScDoctor pure-Python fallback paths

"""Contracts for the ScDoctor pure-Python adapt and Hamming(7,4) ECC fallbacks."""

from __future__ import annotations

import pytest

from sc_neurocore.debug import sc_doctor as doctor_module
from sc_neurocore.debug.sc_doctor import ScDoctor


@pytest.fixture
def python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the pure-Python path by disabling the Rust extension dispatch."""
    monkeypatch.setattr(doctor_module, "_HAS_RUST_DOCTOR", False)
    monkeypatch.setattr(doctor_module, "_sdc", None)


def test_adapt_high_correlation_doubles_and_enables_ecc(python_fallback: None) -> None:
    """High correlation doubles the bitstream and enables ECC once it passes 2048."""
    doctor = ScDoctor(2048)

    doctor.adapt(0.2)

    assert doctor.current_bitstream_length == 4096
    assert doctor.error_correction_enabled is True


def test_adapt_low_correlation_halves_and_disables_ecc(python_fallback: None) -> None:
    """Low correlation halves the bitstream and clears ECC down to the 256 floor."""
    doctor = ScDoctor(512)
    doctor.error_correction_enabled = True

    doctor.adapt(0.01)

    assert doctor.current_bitstream_length == 256
    assert doctor.error_correction_enabled is False


def test_adapt_holds_at_floor_for_low_correlation(python_fallback: None) -> None:
    """At the 256 floor a low correlation leaves the bitstream length unchanged."""
    doctor = ScDoctor(256)

    doctor.adapt(0.01)

    assert doctor.current_bitstream_length == 256


def test_ecc_bypass_returns_low_nibble_when_disabled(python_fallback: None) -> None:
    """With ECC disabled both codecs pass the low four bits through unchanged."""
    doctor = ScDoctor()

    assert doctor.encode_ecc(0b1011) == 0b1011
    assert doctor.decode_ecc(0b1111011) == 0b1011


def test_hamming74_python_roundtrip_for_all_nibbles(python_fallback: None) -> None:
    """The pure-Python Hamming(7,4) codec round-trips every 4-bit value exactly."""
    doctor = ScDoctor()
    doctor.error_correction_enabled = True

    for nibble in range(16):
        assert doctor.decode_ecc(doctor.encode_ecc(nibble)) == nibble


def test_hamming74_python_corrects_every_single_bit_error(python_fallback: None) -> None:
    """The pure-Python decoder corrects any single-bit error in the 7-bit codeword."""
    doctor = ScDoctor()
    doctor.error_correction_enabled = True

    for nibble in range(16):
        codeword = doctor.encode_ecc(nibble)
        for bit in range(7):
            assert doctor.decode_ecc(codeword ^ (1 << bit)) == nibble


@pytest.mark.skipif(
    not doctor_module._HAS_RUST_DOCTOR, reason="Rust stochastic_doctor_core not built"
)
def test_python_fallback_is_bit_exact_with_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python fallback reproduces the Rust extension bit-for-bit, as documented."""
    rust_doctor = ScDoctor()
    rust_doctor.error_correction_enabled = True
    rust_encoded = [rust_doctor.encode_ecc(nibble) for nibble in range(16)]

    monkeypatch.setattr(doctor_module, "_HAS_RUST_DOCTOR", False)
    monkeypatch.setattr(doctor_module, "_sdc", None)
    python_doctor = ScDoctor()
    python_doctor.error_correction_enabled = True

    assert [python_doctor.encode_ecc(nibble) for nibble in range(16)] == rust_encoded
    assert [python_doctor.decode_ecc(code) for code in rust_encoded] == list(range(16))
