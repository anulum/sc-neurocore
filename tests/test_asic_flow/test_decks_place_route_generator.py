# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlaceRouteGenerator from former test_decks.py

"""Focused suite: TestPlaceRouteGenerator from former test_decks.py."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    DesignParams,
    PDKConfig,
    PDKType,
    PlaceRouteGenerator,
)


class TestPlaceRouteGenerator:
    """Place-and-route deck generation."""

    def test_generates_tcl(self) -> None:
        """The deck places, builds the clock tree and routes."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "global_placement" in tcl
        assert "detailed_route" in tcl
        assert "clock_tree_synthesis" in tcl

    def test_utilisation(self) -> None:
        """The deck carries the design's utilisation of 0.60."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(utilisation=0.6)
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "0.60" in tcl

    def test_cell_prefix(self) -> None:
        """The deck uses the PDK's cell prefix."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert pdk.cell_prefix in tcl
