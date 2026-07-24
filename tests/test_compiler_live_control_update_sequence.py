# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (update_sequence) from former test_compiler_live_control.py

from __future__ import annotations

from tests.compiler_live_control_support import *  # noqa: F403

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


