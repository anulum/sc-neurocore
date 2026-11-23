# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIRDropGenerator from former test_constraints.py

"""Focused suite: TestIRDropGenerator from former test_constraints.py."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    DesignParams,
    IRDropGenerator,
    PDKConfig,
    PDKType,
)


class TestIRDropGenerator:
    """IR-drop analysis scripts."""

    def test_generates_script(self) -> None:
        """The script analyses the VDD grid at the given toggle rate."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = IRDropGenerator.generate(pdk, design, toggle_rate=0.15)
        assert "analyze_power_grid" in script
        assert "0.150" in script
        assert "VDD" in script
