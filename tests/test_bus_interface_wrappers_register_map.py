# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegisterMap from former test_bus_interface_wrappers.py

"""Focused suite: TestRegisterMap from former test_bus_interface_wrappers.py."""

from __future__ import annotations

from tests.bus_interface_wrappers_support import *  # noqa: F403


class TestRegisterMap:
    """Test register map generation."""

    def test_standard_layout(self) -> None:
        """Standard registers should be at expected offsets."""
        rmap = generate_register_map(LIF_PARAMS)
        assert rmap["CTRL"] == 0
        assert rmap["I_T"] == 4
        assert rmap["SPIKE_COUNT"] == 8
        assert rmap["P_V_REST"] == 12

    def test_custom_base_address(self) -> None:
        """Base address should shift all registers."""
        rmap = generate_register_map(LIF_PARAMS, base_address=0x1000)
        assert rmap["CTRL"] == 0x1000
        assert rmap["I_T"] == 0x1004

    def test_invalid_bus(self) -> None:
        """Should raise on invalid bus protocol."""
        with pytest.raises(ValueError, match="Unsupported bus"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus=cast(BusProtocol, "spi"))
