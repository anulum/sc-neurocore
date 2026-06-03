# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Contract tests for compiler live-control specifications."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.live_control import (
    MMIOUpdateSpec,
    ParameterBankSpec,
    TrapSpec,
)


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


def test_parameter_bank_overlap_detected_by_mmio_contract() -> None:
    first = ParameterBankSpec(
        bank_name="a",
        start_address_bytes=0x4000,
        parameter_count=2,
        parameter_names=("a_0", "a_1"),
    )
    second = ParameterBankSpec(
        bank_name="b",
        start_address_bytes=0x4002,
        parameter_count=2,
        parameter_names=("b_0", "b_1"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        MMIOUpdateSpec(
            bus_protocol="axi_lite",
            banks=(first, second),
        )


def test_trap_spec_offset_aligned_validation() -> None:
    with pytest.raises(ValueError, match="4-byte aligned"):
        TrapSpec(flag_register_offset=3)


def test_mmio_update_serialization_roundtrip() -> None:
    bank = ParameterBankSpec(
        bank_name="control",
        start_address_bytes=0x5000,
        parameter_count=1,
        parameter_names=("k_mag",),
    )
    trap = TrapSpec(action="interrupt", flag_register_offset=0x20, max_flags=4)
    original = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        read_data_width=32,
        write_data_width=32,
        address_width_bits=32,
        bank_name_width=32,
        supports_burst=True,
        supports_partial_write=False,
        trap=trap,
    )

    payload = original.to_dict()
    restored = MMIOUpdateSpec.from_dict(payload)
    assert restored == original
    assert restored.has_traps is True


def test_mmio_update_protocol_alias_and_width_guards() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x6000,
        parameter_count=1,
        parameter_names=("w_0",),
    )

    spec = MMIOUpdateSpec(
        bus_protocol="AXI_LITE",
        banks=(bank,),
        write_data_width=16,
    )
    assert spec.bus_protocol == "axi4_lite"
    assert spec.total_address_space_bytes == bank.span_bytes


def test_mmio_rejects_invalid_bus_protocol() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x7000,
        parameter_count=1,
        parameter_names=("w_0",),
    )
    with pytest.raises(ValueError, match="Unsupported MMIO protocol"):
        MMIOUpdateSpec(bus_protocol="apb", banks=(bank,))


def test_mmio_rejects_duplicate_bank_name() -> None:
    a = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x8000,
        parameter_count=1,
        parameter_names=("w_0",),
    )
    b = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x8010,
        parameter_count=1,
        parameter_names=("k_0",),
    )

    with pytest.raises(ValueError, match="bank names"):
        MMIOUpdateSpec(bus_protocol="pcie", banks=(a, b))
