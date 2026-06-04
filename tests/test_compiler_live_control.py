# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for compiler live-control contract schema

"""Contract tests for compiler live-control specifications."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.live_control import (
    CONTROL_COMMIT,
    CONTROL_REGISTER_SPAN_BYTES,
    CONTROL_UPDATE_VALID,
    MMIOUpdateSpec,
    ParameterBankSpec,
    STATUS_TRAP_LATCHED,
    STATUS_UPDATE_ACK,
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
        control_base_address_bytes=0x0,
        trap=trap,
    )

    payload = original.to_dict()
    restored = MMIOUpdateSpec.from_dict(payload)
    assert restored == original
    assert restored.has_traps is True
    assert payload["control_registers"]["control"] == 0x0
    assert payload["status_bits"]["trap_latched"] == STATUS_TRAP_LATCHED


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
    assert spec.total_address_space_bytes == 0x6002


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


def test_mmio_update_sequence_stages_and_commits_wide_bfp_word() -> None:
    bank = ParameterBankSpec(
        bank_name="bfp_weights",
        start_address_bytes=0x9000,
        parameter_count=2,
        parameter_names=("w_0", "w_1"),
        precision_mode="bfp",
        bfp_exponent_bits=12,
        bfp_mantissa_bits=36,
    )
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
    )

    writes = spec.build_update_sequence("bfp_weights", "w_1", 0x1234_5678_9ABC)

    assert [write.purpose for write in writes] == [
        "select_bank",
        "select_entry",
        "write_data_lo",
        "write_data_hi",
        "commit_update",
    ]
    assert writes[0].address_bytes == 0x108
    assert writes[0].value == 0
    assert writes[1].address_bytes == 0x10C
    assert writes[1].value == 1
    assert writes[2].value == 0x5678_9ABC
    assert writes[3].value == 0x1234
    assert writes[-1].address_bytes == 0x100
    assert writes[-1].value == CONTROL_UPDATE_VALID | CONTROL_COMMIT
    assert spec.status_bits["update_ack"] == STATUS_UPDATE_ACK
    assert CONTROL_REGISTER_SPAN_BYTES == 0x20


def test_mmio_update_sequence_rejects_read_only_and_unknown_entries() -> None:
    bank = ParameterBankSpec(
        bank_name="coefficients",
        start_address_bytes=0xA000,
        parameter_count=1,
        parameter_names=("k_mag",),
        writable=False,
    )
    spec = MMIOUpdateSpec(bus_protocol="pcie", banks=(bank,), control_base_address_bytes=0x100)

    with pytest.raises(ValueError, match="read-only"):
        spec.build_update_sequence("coefficients", "k_mag", 0x10)
    with pytest.raises(ValueError, match="unknown parameter bank"):
        spec.build_update_sequence("weights", 0, 0x10)


def test_mmio_trap_clear_sequence_requires_enabled_traps() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0xB000,
        parameter_count=1,
        parameter_names=("w_0",),
    )
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
        trap=TrapSpec(enabled=True, max_flags=8),
    )

    writes = spec.build_trap_clear_sequence()
    assert [write.purpose for write in writes] == ["clear_trap", "clear_trap"]
    assert writes[0].address_bytes == 0x11C
    assert writes[0].value == 8

    disabled = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
        trap=TrapSpec(enabled=False),
    )
    with pytest.raises(ValueError, match="enabled traps"):
        disabled.build_trap_clear_sequence()
