# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFloorplanGenerator from former test_decks.py

"""Focused suite: TestFloorplanGenerator from former test_decks.py."""

from __future__ import annotations

from tests.test_asic_flow.decks_support import *  # noqa: F403


class TestFloorplanGenerator:
    def test_generates_tcl(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = FloorplanGenerator.generate(pdk, design)
        assert "initialize_floorplan" in tcl
        assert "read_lef" in tcl

    def test_power_grid(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(power_nets=["VDD", "VSS"])
        tcl = FloorplanGenerator.generate(pdk, design)
        assert "VDD" in tcl
        assert "VSS" in tcl

    def test_single_power_net_omits_two_net_power_grid(self) -> None:
        """A one-net design avoids emitting an invalid power/ground grid pair."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(power_nets=["VDD"])

        tcl = FloorplanGenerator.generate(pdk, design)

        assert "initialize_floorplan" in tcl
        assert "define_pdn_grid" not in tcl
