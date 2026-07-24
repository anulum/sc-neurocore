# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (mmio_contract) from former test_compiler_live_control.py

from __future__ import annotations

from tests.compiler_live_control_support import *  # noqa: F403

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


