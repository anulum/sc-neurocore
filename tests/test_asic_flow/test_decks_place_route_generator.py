# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlaceRouteGenerator from former test_decks.py

"""Focused suite: TestPlaceRouteGenerator from former test_decks.py."""

from __future__ import annotations

from tests.test_asic_flow.decks_support import *  # noqa: F403


class TestPlaceRouteGenerator:
    def test_generates_tcl(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "global_placement" in tcl
        assert "detailed_route" in tcl
        assert "clock_tree_synthesis" in tcl

    def test_utilisation(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(utilisation=0.6)
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "0.60" in tcl

    def test_cell_prefix(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert pdk.cell_prefix in tcl
