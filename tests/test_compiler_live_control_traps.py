# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (traps) from former test_compiler_live_control.py

from __future__ import annotations

from tests.compiler_live_control_support import *  # noqa: F403


def test_trap_spec_offset_aligned_validation() -> None:
    with pytest.raises(ValueError, match="4-byte aligned"):
        TrapSpec(flag_register_offset=3)


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

    writes = spec.build_selective_trap_clear_sequence(TRAP_STAGED_OVERFLOW | TRAP_PARTIAL_WRITE)

    assert [write.purpose for write in writes] == ["clear_trap", "clear_trap"]
    assert writes[0].address_bytes == 0x11C
    assert writes[0].value == TRAP_STAGED_OVERFLOW | TRAP_PARTIAL_WRITE
    assert writes[1].address_bytes == 0x100
    assert writes[1].value == CONTROL_CLEAR_TRAP

    with pytest.raises(ValueError, match="trap_mask"):
        spec.build_selective_trap_clear_sequence(True)
    with pytest.raises(ValueError, match="host-visible trap bits"):
        spec.build_selective_trap_clear_sequence(spec.trap_clear_mask + 1)
