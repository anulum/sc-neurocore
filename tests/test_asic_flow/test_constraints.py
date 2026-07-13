# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC auxiliary constraint tests

"""Exercise CDC, IR-drop, IO-placement, and equivalence decks."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    CDCCheckGenerator,
    DesignParams,
    IOConstraintGenerator,
    IOPin,
    IRDropGenerator,
    LECGenerator,
    PDKConfig,
    PDKType,
)


class TestCDCCheckGenerator:
    def test_single_domain(self) -> None:
        design = DesignParams(clock_name="clk")
        script = CDCCheckGenerator.generate(design)
        assert "clk" in script
        assert "report_cdc" in script

    def test_multi_domain(self) -> None:
        design = DesignParams()
        script = CDCCheckGenerator.generate(design, clock_domains=["clk_fast", "clk_slow"])
        assert "clk_fast" in script
        assert "clk_slow" in script


class TestIRDropGenerator:
    def test_generates_script(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = IRDropGenerator.generate(pdk, design, toggle_rate=0.15)
        assert "analyze_power_grid" in script
        assert "0.150" in script
        assert "VDD" in script


class TestIOConstraints:
    def test_generate(self) -> None:
        pins = [IOPin("clk", "input", "N"), IOPin("data_out", "output", "S")]
        design = DesignParams()
        script = IOConstraintGenerator.generate(pins, design)
        assert "clk" in script
        assert "data_out" in script

    def test_auto_assign(self) -> None:
        names = ["a", "b", "c", "d", "e"]
        pins = IOConstraintGenerator.auto_assign(names)
        assert len(pins) == 5
        sides = set(p.side for p in pins)
        assert len(sides) == 4


class TestLECGenerator:
    def test_generates_lec(self) -> None:
        design = DesignParams(top_module="sc_lif")
        script = LECGenerator.generate(design)
        assert "equiv_make" in script
        assert "equiv_status" in script
        assert "sc_lif" in script
