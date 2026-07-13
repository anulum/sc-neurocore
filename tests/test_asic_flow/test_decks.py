# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC deck generator tests

"""Exercise Yosys, OpenROAD, SDC, and GDSII deck generation."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    DesignParams,
    FloorplanGenerator,
    GDSIIExporter,
    PDKConfig,
    PDKType,
    PlaceRouteGenerator,
    SCASICOptimisationConfig,
    SDCGenerator,
    SynthesisGenerator,
)


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


class TestSDCGenerator:
    def test_generates_sdc(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        sdc = SDCGenerator.generate(pdk, design)
        assert "create_clock" in sdc
        assert design.clock_name in sdc

    def test_clock_period(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=100.0)
        sdc = SDCGenerator.generate(pdk, design)
        assert "10.000" in sdc

    def test_false_path_reset(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(reset_name="rst_n")
        sdc = SDCGenerator.generate(pdk, design)
        assert "rst_n" in sdc
        assert "false_path" in sdc

    def test_sc_fanout_constraint(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(sc_optimisation=SCASICOptimisationConfig(max_fanout=8))
        sdc = SDCGenerator.generate(pdk, design)
        assert "set_max_fanout 8" in sdc


class TestGDSIIExporter:
    def test_open_source_export(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert ".gds" in script
        assert design.top_module in script

    def test_commercial_export(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.INTEL16)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert "vendor" in script.lower()
