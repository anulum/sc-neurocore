# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (parameter_bank) from former test_compiler_live_control.py

from __future__ import annotations

from tests.compiler_live_control_support import *  # noqa: F403

def test_parameter_bank_q88_bounds_and_widths() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x1000,
        parameter_count=4,
        parameter_names=("w_0", "w_1", "w_2", "w_3"),
        precision_mode="q",
        q_format="Q8.8",
    )

    assert bank.entry_width_bits == 16
    assert bank.entry_width_bytes == 2
    assert bank.span_bytes == 8
    assert bank.end_address_bytes == 0x1008


def test_parameter_bank_rejects_invalid_q_format() -> None:
    with pytest.raises(ValueError, match="Expected format"):
        ParameterBankSpec(
            bank_name="weights",
            start_address_bytes=0x0,
            parameter_count=1,
            parameter_names=("w_0",),
            precision_mode="q",
            q_format="bad",
        )


def test_parameter_bank_supports_q16_16() -> None:
    bank = ParameterBankSpec(
        bank_name="kuramoto",
        start_address_bytes=0x2000,
        parameter_count=2,
        parameter_names=("k", "theta"),
        precision_mode="q",
        q_format="Q16.16",
    )
    assert bank.entry_width_bits == 32
    assert bank.entry_width_bytes == 4
    assert bank.entry_address("theta") == 0x2004
    assert bank.normalise_encoded_word(-1) == 0xFFFF_FFFF


def test_parameter_bank_supports_bfp() -> None:
    bank = ParameterBankSpec(
        bank_name="bfp_weights",
        start_address_bytes=0x3000,
        parameter_count=2,
        parameter_names=("w_0", "w_1"),
        precision_mode="bfp",
        bfp_exponent_bits=6,
        bfp_mantissa_bits=10,
    )

    assert bank.entry_width_bits == 16
    assert bank.entry_width_bytes == 2
    assert bank.entry_address(1) == 0x3002


def test_parameter_bank_rejects_out_of_range_encoded_word() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x3000,
        parameter_count=1,
        parameter_names=("w_0",),
        precision_mode="q",
        q_format="Q8.8",
    )

    with pytest.raises(ValueError, match="entry width"):
        bank.normalise_encoded_word(1 << 16)
    with pytest.raises(ValueError, match="entry width"):
        bank.normalise_encoded_word(-(1 << 15) - 1)


def test_parameter_bank_overlap_detected_by_mmio_contract() -> None:
    first = ParameterBankSpec(
        bank_name="a",
        start_address_bytes=0x4000,
        parameter_count=4,
        parameter_names=("a_0", "a_1", "a_2", "a_3"),
    )
    second = ParameterBankSpec(
        bank_name="b",
        start_address_bytes=0x4004,
        parameter_count=2,
        parameter_names=("b_0", "b_1"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        MMIOUpdateSpec(
            bus_protocol="axi_lite",
            banks=(first, second),
        )


def test_mmio_control_window_must_not_overlap_parameter_banks() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x10,
        parameter_count=4,
        parameter_names=("w_0", "w_1", "w_2", "w_3"),
    )

    with pytest.raises(ValueError, match="Control register window"):
        MMIOUpdateSpec(
            bus_protocol="axi4_lite",
            banks=(bank,),
            control_base_address_bytes=0x0,
        )


