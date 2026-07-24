# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (readback_sequence) from former test_compiler_live_control.py

from __future__ import annotations

from tests.compiler_live_control_support import *  # noqa: F403

def test_mmio_readback_sequence_selects_committed_q16_entry() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x9000,
        parameter_count=2,
        parameter_names=("w_0", "w_1"),
        q_format="Q16.16",
    )
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
    )

    ops = spec.build_readback_sequence("weights", "w_1")

    assert [op.purpose for op in ops] == [
        "select_bank",
        "select_entry",
        "read_active_data_lo",
    ]
    assert ops[0].address_bytes == 0x108
    assert ops[0].value == 0
    assert ops[1].address_bytes == 0x10C
    assert ops[1].value == 1
    assert ops[2].address_bytes == 0x124
    assert ops[2].width_bits == 32


def test_mmio_readback_sequence_reads_wide_bfp_high_word() -> None:
    bank = ParameterBankSpec(
        bank_name="bfp_weights",
        start_address_bytes=0x9000,
        parameter_count=1,
        parameter_names=("w_0",),
        precision_mode="bfp",
        bfp_exponent_bits=12,
        bfp_mantissa_bits=36,
        writable=False,
    )
    spec = MMIOUpdateSpec(
        bus_protocol="pcie",
        banks=(bank,),
        control_base_address_bytes=0x100,
    )

    ops = spec.build_readback_sequence("bfp_weights", "w_0")

    assert [op.purpose for op in ops] == [
        "select_bank",
        "select_entry",
        "read_active_data_lo",
        "read_active_data_hi",
    ]
    assert ops[2].address_bytes == 0x124
    assert ops[3].address_bytes == 0x128
    assert ops[3].width_bits == 32


def test_mmio_readback_sequence_rejects_unknown_entries() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x9000,
        parameter_count=1,
        parameter_names=("w_0",),
    )
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        spec.build_readback_sequence("weights", "w_1")


