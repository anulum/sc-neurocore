# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSynthesisGenerator from former test_decks.py

"""Focused suite: TestSynthesisGenerator from former test_decks.py."""

from __future__ import annotations

from tests.test_asic_flow.decks_support import *  # noqa: F403


class TestSynthesisGenerator:
    def test_generates_tcl(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "yosys" in tcl.lower() or "synth" in tcl.lower()
        assert design.top_module in tcl

    def test_includes_liberty(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert pdk.liberty_file in tcl

    def test_includes_rtl_files(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(rtl_files=["top.v", "neuron.v"])
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "top.v" in tcl
        assert "neuron.v" in tcl

    def test_clock_constraint(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=200.0)
        tcl = SynthesisGenerator.generate(pdk, design)
        # 200 MHz = 5ns; SC optimisation uses the default 90% delay margin.
        assert "4500" in tcl

    def test_sc_optimisation_passes_enabled(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "wreduce" in tcl
        assert "opt_share" in tcl
        assert "keep_hierarchy" in tcl

    def test_sc_optimisation_can_disable_counter_sharing(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(
            sc_optimisation=SCASICOptimisationConfig(share_stochastic_counters=False)
        )
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "opt_share" not in tcl

    def test_sc_optimisation_can_expose_lfsr_hierarchy(self) -> None:
        """Disabling hierarchy preservation removes the Yosys keep attribute."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(
            sc_optimisation=SCASICOptimisationConfig(preserve_lfsr_hierarchy=False)
        )

        assert "keep_hierarchy" not in SynthesisGenerator.generate(pdk, design)
