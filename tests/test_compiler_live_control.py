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
    CONTROL_CLEAR_TRAP,
    CONTROL_REGISTER_SPAN_BYTES,
    CONTROL_ROLLBACK,
    CONTROL_UPDATE_VALID,
    MMIOUpdateSpec,
    ParameterBankSpec,
    STATUS_APPLIED,
    STATUS_CHECKSUM_VALID,
    STATUS_ROLLBACK_ACK,
    STATUS_SHADOW_LOADED,
    STATUS_TRAP_LATCHED,
    STATUS_UPDATE_ACK,
    TRAP_CHECKSUM_MISMATCH,
    TRAP_INVALID_SELECTION,
    TRAP_PARTIAL_WRITE,
    TRAP_READ_ONLY_BANK,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
    TrapSpec,
    UPDATE_CHECKSUM_ALGORITHM,
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
    assert payload["control_registers"]["write_checksum"] == 0x20
    assert payload["checksum_algorithm"] == UPDATE_CHECKSUM_ALGORITHM
    assert payload["status_bits"]["trap_latched"] == STATUS_TRAP_LATCHED
    assert payload["status_bits"]["shadow_loaded"] == STATUS_SHADOW_LOADED
    assert payload["status_bits"]["applied"] == STATUS_APPLIED
    assert payload["status_bits"]["rollback_ack"] == STATUS_ROLLBACK_ACK
    assert payload["status_bits"]["checksum_valid"] == STATUS_CHECKSUM_VALID
    assert payload["trap_bits"]["staged_overflow"] == TRAP_STAGED_OVERFLOW
    assert payload["trap_bits"]["staged_underflow"] == TRAP_STAGED_UNDERFLOW
    assert payload["trap_bits"]["checksum_mismatch"] == TRAP_CHECKSUM_MISMATCH
    assert payload["trap_bits"]["invalid_selection"] == TRAP_INVALID_SELECTION
    assert payload["trap_bits"]["read_only_bank"] == TRAP_READ_ONLY_BANK
    assert payload["trap_bits"]["partial_write"] == TRAP_PARTIAL_WRITE
    assert payload["effective_trap_width"] == 6
    assert payload["trap_clear_mask"] == 0x3F


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
        "write_checksum",
        "load_shadow",
        "apply_shadow",
    ]
    assert writes[0].address_bytes == 0x108
    assert writes[0].value == 0
    assert writes[1].address_bytes == 0x10C
    assert writes[1].value == 1
    assert writes[2].value == 0x5678_9ABC
    assert writes[3].value == 0x1234
    assert writes[4].address_bytes == 0x120
    assert writes[4].value == spec.update_checksum("bfp_weights", "w_1", 0x1234_5678_9ABC)
    assert writes[-2].address_bytes == 0x100
    assert writes[-2].value == CONTROL_UPDATE_VALID
    assert writes[-1].address_bytes == 0x100
    assert writes[-1].value == CONTROL_COMMIT
    assert spec.status_bits["update_ack"] == STATUS_UPDATE_ACK
    assert CONTROL_REGISTER_SPAN_BYTES == 0x2C


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


def test_mmio_update_checksum_uses_ieee_crc32_update_guard() -> None:
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

    checksum = spec.update_checksum("weights", "w_0", 0x1234)

    assert checksum == 0x1D7D9B35
    assert checksum != 0x00001234
    assert spec.update_checksum("weights", "w_1", 0x1234) != checksum
    assert spec.update_checksum("weights", "w_0", 0x1235) != checksum
    assert spec.update_checksum("weights", "w_0", 0x00010000) == 0x27E798F0


def test_mmio_update_sequence_zeroes_high_word_for_narrow_entries() -> None:
    bank = ParameterBankSpec(
        bank_name="weights",
        start_address_bytes=0x9000,
        parameter_count=1,
        parameter_names=("w_0",),
        q_format="Q8.8",
    )
    spec = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
    )

    writes = spec.build_update_sequence("weights", "w_0", 0x1234)

    assert [write.purpose for write in writes] == [
        "select_bank",
        "select_entry",
        "write_data_lo",
        "write_data_hi",
        "write_checksum",
        "load_shadow",
        "apply_shadow",
    ]
    assert writes[2].address_bytes == 0x110
    assert writes[2].value == 0x1234
    assert writes[3].address_bytes == 0x114
    assert writes[3].value == 0
    assert writes[4].value == spec.update_checksum("weights", "w_0", 0x1234)


def test_mmio_update_sequence_supports_explicit_apply_and_rollback() -> None:
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

    apply_writes = spec.build_apply_sequence()
    rollback_writes = spec.build_rollback_sequence()

    assert apply_writes[0].purpose == "apply_shadow"
    assert apply_writes[0].value == CONTROL_COMMIT
    assert rollback_writes[0].purpose == "rollback_shadow"
    assert rollback_writes[0].value == CONTROL_ROLLBACK
    assert spec.control_bits["rollback"] == CONTROL_ROLLBACK


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
    assert writes[0].value == 0xFF

    disabled = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
        trap=TrapSpec(enabled=False),
    )
    with pytest.raises(ValueError, match="enabled traps"):
        disabled.build_trap_clear_sequence()

    narrow = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        banks=(bank,),
        control_base_address_bytes=0x100,
        trap=TrapSpec(enabled=True, max_flags=1),
    )
    assert narrow.effective_trap_width == 6
    assert narrow.trap_clear_mask == 0x3F
    assert narrow.build_trap_clear_sequence()[0].value == 0x3F


def test_mmio_selective_trap_clear_sequence_preserves_unselected_faults() -> None:
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

    writes = spec.build_selective_trap_clear_sequence(
        TRAP_STAGED_OVERFLOW | TRAP_PARTIAL_WRITE
    )

    assert [write.purpose for write in writes] == ["clear_trap", "clear_trap"]
    assert writes[0].address_bytes == 0x11C
    assert writes[0].value == TRAP_STAGED_OVERFLOW | TRAP_PARTIAL_WRITE
    assert writes[1].address_bytes == 0x100
    assert writes[1].value == CONTROL_CLEAR_TRAP

    with pytest.raises(ValueError, match="trap_mask"):
        spec.build_selective_trap_clear_sequence(True)
    with pytest.raises(ValueError, match="host-visible trap bits"):
        spec.build_selective_trap_clear_sequence(spec.trap_clear_mask + 1)
