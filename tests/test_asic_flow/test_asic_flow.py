# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — OpenROAD ASIC Tape-Out Flow Tests

import json
from pathlib import Path

import pytest

from sc_neurocore.asic_flow.asic_flow import (
    ASICFlowBundle,
    ASICFlowGenerator,
    ASICFlowOutput,
    BlockConfig,
    CDCCheckGenerator,
    CornerType,
    DRCViolation,
    DesignParams,
    FloorplanGenerator,
    GDSIIExporter,
    HierarchicalFlow,
    IOConstraintGenerator,
    IOPin,
    IRDropGenerator,
    LECGenerator,
    MultiCornerAnalysis,
    OCVConfig,
    OpenSourcePDKResolver,
    PDKConfig,
    PDKResolution,
    PDKType,
    PVTCorner,
    PlaceRouteGenerator,
    PreSynthEstimator,
    ResolvedPDKFiles,
    SCASICOptimisationConfig,
    SDCGenerator,
    SignoffCheckResult,
    SignoffGenerator,
    SignoffSummary,
    SynthesisGenerator,
    TapeOutChecklist,
    validate_pdk_installation,
    validate_pdk,
    generate_asic_flow_bundle,
    _normalise_pdk_type,
)


# ── PDKConfig Tests ──────────────────────────────────────────────────


class TestPDKConfig:
    def test_sky130_preset(self):
        cfg = PDKConfig.from_pdk_type(PDKType.SKY130)
        assert "sky130" in cfg.liberty_file
        assert cfg.voltage_v == 1.8
        assert cfg.min_feature_nm == 130

    def test_gf180_preset(self):
        cfg = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        assert "gf180" in cfg.liberty_file
        assert cfg.tech_lef.endswith(".tlef")
        assert cfg.voltage_v == 3.3

    def test_tsmc28_preset(self):
        cfg = PDKConfig.from_pdk_type(PDKType.TSMC28)
        assert cfg.min_feature_nm == 28
        assert cfg.metal_layers == 10

    def test_intel16_preset(self):
        cfg = PDKConfig.from_pdk_type(PDKType.INTEL16)
        assert cfg.min_feature_nm == 16

    def test_custom_preset(self):
        cfg = PDKConfig.from_pdk_type(PDKType.CUSTOM)
        assert cfg.liberty_file == ""

    def test_is_open_source(self):
        assert PDKConfig.from_pdk_type(PDKType.SKY130).is_open_source is True
        assert PDKConfig.from_pdk_type(PDKType.TSMC28).is_open_source is False

    def test_all_pdks(self):
        for pdk in PDKType:
            cfg = PDKConfig.from_pdk_type(pdk)
            assert cfg.min_feature_nm > 0

    def test_bind_pdk_root(self):
        cfg = PDKConfig.from_pdk_type(PDKType.SKY130).with_pdk_root("/opt/pdk")
        assert cfg.liberty_file.startswith("/opt/pdk/sky130A")
        assert "$PDK_ROOT" not in cfg.lef_file


class TestOpenSourcePDKResolver:
    def test_resolves_sky130_manifest(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root="/opt/pdk")
        assert isinstance(resolution, PDKResolution)
        assert isinstance(resolution.files, ResolvedPDKFiles)
        assert resolution.pdk.liberty_file.startswith("/opt/pdk/sky130A")
        assert "sky130.lydrc" in resolution.files.drc_deck

    def test_resolves_gf180_manifest(self):
        pdk = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root="/opt/pdk")
        assert resolution.pdk.tech_lef.endswith(".tlef")
        assert "gf180mcuD_setup.tcl" in resolution.files.lvs_setup

    def test_reports_missing_required_files(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(
            pdk, pdk_root="/definitely/missing/pdk", require_existing=True
        )
        assert not resolution.usable_for_synthesis
        assert "liberty_file" in resolution.missing_required

    def test_accepts_minimal_existing_synthesis_files(self, tmp_path):
        root = tmp_path / "pdk"
        paths = [
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-pdk-file\n", encoding="utf-8")

        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root=str(root), require_existing=True)
        assert resolution.usable_for_synthesis
        assert not resolution.usable_for_signoff


# ── DesignParams Tests ───────────────────────────────────────────────


class TestDesignParams:
    def test_clock_period(self):
        dp = DesignParams(target_frequency_mhz=100.0)
        assert abs(dp.clock_period_ns - 10.0) < 0.01

    def test_die_dimensions(self):
        dp = DesignParams(die_area_um=(0, 0, 1000, 800))
        assert dp.die_width_um == 1000
        assert dp.die_height_um == 800

    def test_core_area(self):
        dp = DesignParams(core_area_um=(20, 20, 480, 480))
        assert abs(dp.core_area_mm2 - 0.2116) < 0.001


# ── SynthesisGenerator Tests ─────────────────────────────────────────


class TestSynthesisGenerator:
    def test_generates_tcl(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "yosys" in tcl.lower() or "synth" in tcl.lower()
        assert design.top_module in tcl

    def test_includes_liberty(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert pdk.liberty_file in tcl

    def test_includes_rtl_files(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(rtl_files=["top.v", "neuron.v"])
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "top.v" in tcl
        assert "neuron.v" in tcl

    def test_clock_constraint(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=200.0)
        tcl = SynthesisGenerator.generate(pdk, design)
        # 200 MHz = 5ns; SC optimisation uses the default 90% delay margin.
        assert "4500" in tcl

    def test_sc_optimisation_passes_enabled(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "wreduce" in tcl
        assert "opt_share" in tcl
        assert "keep_hierarchy" in tcl

    def test_sc_optimisation_can_disable_counter_sharing(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(
            sc_optimisation=SCASICOptimisationConfig(share_stochastic_counters=False)
        )
        tcl = SynthesisGenerator.generate(pdk, design)
        assert "opt_share" not in tcl


# ── FloorplanGenerator Tests ─────────────────────────────────────────


class TestFloorplanGenerator:
    def test_generates_tcl(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = FloorplanGenerator.generate(pdk, design)
        assert "initialize_floorplan" in tcl
        assert "read_lef" in tcl

    def test_power_grid(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(power_nets=["VDD", "VSS"])
        tcl = FloorplanGenerator.generate(pdk, design)
        assert "VDD" in tcl
        assert "VSS" in tcl


# ── PlaceRouteGenerator Tests ────────────────────────────────────────


class TestPlaceRouteGenerator:
    def test_generates_tcl(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "global_placement" in tcl
        assert "detailed_route" in tcl
        assert "clock_tree_synthesis" in tcl

    def test_utilisation(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(utilisation=0.6)
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert "0.60" in tcl

    def test_cell_prefix(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        tcl = PlaceRouteGenerator.generate(pdk, design)
        assert pdk.cell_prefix in tcl


# ── SDCGenerator Tests ───────────────────────────────────────────────


class TestSDCGenerator:
    def test_generates_sdc(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        sdc = SDCGenerator.generate(pdk, design)
        assert "create_clock" in sdc
        assert design.clock_name in sdc

    def test_clock_period(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=100.0)
        sdc = SDCGenerator.generate(pdk, design)
        assert "10.000" in sdc

    def test_false_path_reset(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(reset_name="rst_n")
        sdc = SDCGenerator.generate(pdk, design)
        assert "rst_n" in sdc
        assert "false_path" in sdc

    def test_sc_fanout_constraint(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(sc_optimisation=SCASICOptimisationConfig(max_fanout=8))
        sdc = SDCGenerator.generate(pdk, design)
        assert "set_max_fanout 8" in sdc


# ── SignoffGenerator Tests ───────────────────────────────────────────


class TestSignoffGenerator:
    def test_sta_script(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_sta_script(pdk, design)
        assert "report_checks" in script
        assert "report_tns" in script

    def test_drc_open_source(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "klayout" in script.lower() or "KLayout" in script

    def test_drc_commercial(self):
        pdk = PDKConfig.from_pdk_type(PDKType.TSMC28)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "vendor" in script.lower()

    def test_lvs_open_source(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_lvs_script(pdk, design)
        assert "netgen" in script.lower()

    def test_timing_pass(self):
        result = SignoffGenerator.evaluate_timing(wns=0.5, tns=0.0, clock_period_ns=10.0)
        assert result.passed is True
        assert result.check_name == "STA"

    def test_timing_fail(self):
        result = SignoffGenerator.evaluate_timing(wns=-0.3, tns=-5.0, clock_period_ns=10.0)
        assert result.passed is False

    def test_power_pass(self):
        result = SignoffGenerator.evaluate_power(3.0, 0.5, 10.0)
        assert result.passed is True

    def test_power_fail(self):
        result = SignoffGenerator.evaluate_power(8.0, 3.0, 10.0)
        assert result.passed is False

    def test_area_pass(self):
        result = SignoffGenerator.evaluate_area(5000, 150000, 250000)
        assert result.passed is True

    def test_area_fail(self):
        result = SignoffGenerator.evaluate_area(50000, 240000, 250000)
        assert result.passed is False


# ── GDSIIExporter Tests ──────────────────────────────────────────────


class TestGDSIIExporter:
    def test_open_source_export(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert ".gds" in script
        assert design.top_module in script

    def test_commercial_export(self):
        pdk = PDKConfig.from_pdk_type(PDKType.INTEL16)
        design = DesignParams()
        script = GDSIIExporter.generate(pdk, design)
        assert "vendor" in script.lower()


# ── ASICFlowGenerator Tests ──────────────────────────────────────────


class TestASICFlowGenerator:
    def test_full_flow(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(
            top_module="sc_lif_neuron",
            rtl_files=["sc_lif_neuron.v"],
        )
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        assert isinstance(output, ASICFlowOutput)
        assert "synth" in output.synth_tcl.lower()
        assert "create_clock" in output.sdc
        assert "Makefile" in output.filelist

    def test_all_pdks(self):
        gen = ASICFlowGenerator()
        for pdk_type in PDKType:
            pdk = PDKConfig.from_pdk_type(pdk_type)
            design = DesignParams()
            output = gen.generate(pdk, design)
            assert len(output.filelist) > 0

    def test_makefile(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(top_module="test_module")
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        assert "test_module" in output.makefile
        assert "yosys" in output.makefile
        assert "openroad" in output.makefile

    def test_output_dict(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        d = output.to_dict()
        assert "synth.tcl" in d
        assert "Makefile" in d
        assert len(d) == 9

    def test_one_command_bundle_writes_manifest(self, tmp_path):
        design = DesignParams(top_module="edge_snn", rtl_files=["edge_snn.sv"])
        bundle = generate_asic_flow_bundle(
            tmp_path,
            pdk_type="sky130",
            design=design,
            pdk_root="/opt/pdks",
            n_neurons=32,
            n_synapses=512,
            bitstream_width=128,
            n_aer_ports=8,
        )

        assert isinstance(bundle, ASICFlowBundle)
        assert (tmp_path / "synth.tcl").is_file()
        assert (tmp_path / "Makefile").is_file()
        assert (tmp_path / "asic_flow_manifest.json").is_file()
        assert bundle.estimate.gate_count > 0
        manifest = (tmp_path / "asic_flow_manifest.json").read_text(encoding="utf-8")
        assert '"schema": "sc-neurocore.asic_flow_manifest.v1"' in manifest
        assert '"external_eda_executed": false' in manifest
        assert '"physical_ppa_claim_allowed": false' in manifest
        assert '"formal_evidence_attached": false' in manifest
        assert '"formal_evidence_complete_for_claim": false' in manifest
        assert "edge_snn" in manifest

    def test_one_command_bundle_reports_missing_required_pdk_files(self, tmp_path):
        missing_root = tmp_path / "missing_pdk_root"
        bundle = generate_asic_flow_bundle(
            tmp_path / "out",
            pdk_type=PDKType.GF180MCU,
            pdk_root=str(missing_root),
            require_pdk_files=True,
        )

        assert bundle.pdk_resolution.usable_for_synthesis is False
        assert set(bundle.pdk_resolution.missing_required) == {
            "liberty_file",
            "lef_file",
            "tech_lef",
        }

    def test_one_command_bundle_records_formal_evidence_status(self, tmp_path):
        bundle = generate_asic_flow_bundle(
            tmp_path / "out",
            pdk_type=PDKType.SKY130,
            formal_evidence_artifacts=["formal/sc_top.sby", "formal/report.json"],
        )

        manifest = json.loads(Path(bundle.manifest_path).read_text(encoding="utf-8"))
        assert manifest["formal_evidence"]["attached"] is True
        assert manifest["formal_evidence"]["complete_for_claim"] is True
        assert manifest["formal_evidence"]["artifacts"] == [
            "formal/report.json",
            "formal/sc_top.sby",
        ]


# ── PreSynthEstimator Tests ──────────────────────────────────────────


class TestPreSynthEstimator:
    def test_basic_estimate(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        est = PreSynthEstimator.estimate(
            n_neurons=16,
            n_synapses=256,
            bitstream_width=256,
            n_aer_ports=4,
            pdk=pdk,
        )
        assert est.gate_count > 0
        assert est.area_um2 > 0
        assert est.dynamic_power_mw > 0
        assert est.max_frequency_mhz > 0

    def test_scaling_with_neurons(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        small = PreSynthEstimator.estimate(8, 64, 128, 2, pdk)
        large = PreSynthEstimator.estimate(128, 1024, 256, 16, pdk)
        assert large.gate_count > small.gate_count
        assert large.area_um2 > small.area_um2
        assert large.dynamic_power_mw > small.dynamic_power_mw

    def test_pdk_scaling(self):
        sky = PDKConfig.from_pdk_type(PDKType.SKY130)
        tsmc = PDKConfig.from_pdk_type(PDKType.TSMC28)
        est_sky = PreSynthEstimator.estimate(16, 256, 256, 4, sky)
        est_tsmc = PreSynthEstimator.estimate(16, 256, 256, 4, tsmc)
        # 28nm should have smaller area than 130nm
        assert est_tsmc.area_um2 < est_sky.area_um2

    def test_power_scaling(self):
        sky = PDKConfig.from_pdk_type(PDKType.SKY130)
        tsmc = PDKConfig.from_pdk_type(PDKType.TSMC28)
        est_sky = PreSynthEstimator.estimate(16, 256, 256, 4, sky)
        est_tsmc = PreSynthEstimator.estimate(16, 256, 256, 4, tsmc)
        # Lower voltage at 28nm → less dynamic power
        assert est_tsmc.dynamic_power_mw < est_sky.dynamic_power_mw


# ── Multi-Corner Tests (Gap 1) ────────────────────────────────────────


class TestMultiCornerAnalysis:
    def test_generates_all_corners(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = MultiCornerAnalysis.generate(pdk, design)
        assert "ss_125C" in script
        assert "ff_" in script

    def test_custom_corners(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        corners = [PVTCorner(CornerType.TT, 25.0, 1.8, "_tt_025C_1v80")]
        script = MultiCornerAnalysis.generate(pdk, design, corners)
        assert "tt_25C" in script

    def test_worst_slack(self):
        wns = {"tt": 0.5, "ss": -0.2, "ff": 1.0}
        corner, slack = MultiCornerAnalysis.worst_slack(wns)
        assert corner == "ss"
        assert slack == -0.2

    def test_pvt_corner_label(self):
        c = PVTCorner(CornerType.SS, 125.0, 1.62)
        assert "ss" in c.label
        assert "125" in c.label


# ── CDC Check Tests (Gap 2) ───────────────────────────────────────────


class TestCDCCheckGenerator:
    def test_single_domain(self):
        design = DesignParams(clock_name="clk")
        script = CDCCheckGenerator.generate(design)
        assert "clk" in script
        assert "report_cdc" in script

    def test_multi_domain(self):
        design = DesignParams()
        script = CDCCheckGenerator.generate(design, clock_domains=["clk_fast", "clk_slow"])
        assert "clk_fast" in script
        assert "clk_slow" in script


# ── IR Drop Tests (Gap 3) ─────────────────────────────────────────────


class TestIRDropGenerator:
    def test_generates_script(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = IRDropGenerator.generate(pdk, design, toggle_rate=0.15)
        assert "analyze_power_grid" in script
        assert "0.150" in script
        assert "VDD" in script


# ── IO Constraint Tests (Gap 4) ───────────────────────────────────────


class TestIOConstraints:
    def test_generate(self):
        pins = [IOPin("clk", "input", "N"), IOPin("data_out", "output", "S")]
        design = DesignParams()
        script = IOConstraintGenerator.generate(pins, design)
        assert "clk" in script
        assert "data_out" in script

    def test_auto_assign(self):
        names = ["a", "b", "c", "d", "e"]
        pins = IOConstraintGenerator.auto_assign(names)
        assert len(pins) == 5
        sides = set(p.side for p in pins)
        assert len(sides) == 4  # round-robin over NSEW


# ── LEC Tests (Gap 5) ────────────────────────────────────────────────


class TestLECGenerator:
    def test_generates_lec(self):
        design = DesignParams(top_module="sc_lif")
        script = LECGenerator.generate(design)
        assert "equiv_make" in script
        assert "equiv_status" in script
        assert "sc_lif" in script


# ── OCV Config Tests (Gap 6) ──────────────────────────────────────────


class TestOCVConfig:
    def test_default(self):
        ocv = OCVConfig()
        frag = ocv.generate_sdc_fragment()
        assert "set_timing_derate" in frag
        assert "0.950" in frag

    def test_conservative(self):
        ocv = OCVConfig.conservative()
        assert ocv.data_cell_early < 0.95
        assert ocv.data_cell_late > 1.05


# ── Signoff Summary Tests (Gap 7) ─────────────────────────────────────


class TestSignoffSummary:
    def test_all_pass(self):
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            lvs_match=True,
        )
        assert s.all_pass

    def test_drc_failure(self):
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            drc_violations=[DRCViolation("min_width", 5, "error")],
            lvs_match=True,
        )
        assert not s.drc_clean
        assert not s.all_pass

    def test_to_dict(self):
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, "ok"),
            power=SignoffCheckResult("Power", True, "ok"),
            area=SignoffCheckResult("Area", True, "ok"),
        )
        d = s.to_dict()
        assert "timing" in d
        assert "all_pass" in d


# ── PDK Validation Tests (Gap 8) ──────────────────────────────────────


class TestPDKValidation:
    def test_valid_sky130(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        result = validate_pdk(pdk)
        assert result.valid

    def test_invalid_broken_pdk(self):
        pdk = PDKConfig(pdk_type=PDKType.SKY130, liberty_file="", lef_file="", voltage_v=0.0)
        result = validate_pdk(pdk)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_custom_pdk_no_file_check(self):
        pdk = PDKConfig(
            pdk_type=PDKType.CUSTOM, liberty_file="", voltage_v=1.8, clock_period_ns=10.0
        )
        result = validate_pdk(pdk)
        assert result.valid

    def test_installation_check_reports_missing_pdk(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        result = validate_pdk_installation(pdk, pdk_root="/definitely/missing/pdk")
        assert not result.valid
        assert any("liberty_file not found" in err for err in result.errors)

    def test_installation_check_can_require_signoff_files(self, tmp_path):
        root = tmp_path / "pdk"
        paths = [
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-pdk-file\n", encoding="utf-8")

        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        synth_only = validate_pdk_installation(pdk, pdk_root=str(root))
        signoff = validate_pdk_installation(pdk, pdk_root=str(root), require_signoff=True)
        assert synth_only.valid
        assert synth_only.warnings
        assert not signoff.valid


# ── Hierarchical Flow Tests (Gap 9) ───────────────────────────────────


class TestHierarchicalFlow:
    def test_add_blocks(self):
        hf = HierarchicalFlow(top_design=DesignParams())
        hf.add_block(BlockConfig("neuron_array", DesignParams(top_module="neuron_array")))
        hf.add_block(BlockConfig("router", DesignParams(top_module="aer_router")))
        assert hf.block_names() == ["neuron_array", "router"]

    def test_block_scripts(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        hf = HierarchicalFlow(top_design=DesignParams())
        hf.add_block(BlockConfig("core", DesignParams(top_module="sc_core")))
        scripts = hf.generate_block_scripts(pdk)
        assert "core" in scripts
        assert "synth" in scripts["core"].lower()

    def test_top_integration(self):
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        hf = HierarchicalFlow(top_design=DesignParams(top_module="chip_top"))
        hf.add_block(BlockConfig("mem", DesignParams(), is_hard_macro=True, abstract_lef="mem.lef"))
        tcl = hf.generate_top_integration(pdk)
        assert "mem.lef" in tcl
        assert "chip_top" in tcl


# ── Tape-Out Checklist Tests (Gap 10) ─────────────────────────────────


class TestTapeOutChecklist:
    def test_not_ready_default(self):
        cl = TapeOutChecklist()
        assert not cl.is_tape_out_ready
        assert cl.readiness_score == 0.0

    def test_fully_ready(self):
        cl = TapeOutChecklist(
            synthesis_clean=True,
            timing_met=True,
            power_within_budget=True,
            area_within_limit=True,
            drc_clean=True,
            lvs_clean=True,
            formal_equiv_pass=True,
            cdc_clean=True,
            ir_drop_ok=True,
            esd_reviewed=True,
        )
        assert cl.is_tape_out_ready
        assert cl.readiness_score == 1.0
        assert cl.failing_checks() == []

    def test_partial_readiness(self):
        cl = TapeOutChecklist(synthesis_clean=True, timing_met=True)
        assert cl.readiness_score == 0.2
        assert "drc_clean" in cl.failing_checks()

    def test_from_signoff(self):
        summary = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", False, ""),
            area=SignoffCheckResult("Area", True, ""),
            lvs_match=True,
        )
        cl = TapeOutChecklist()
        cl.from_signoff(summary)
        assert cl.timing_met is True
        assert cl.power_within_budget is False
        assert cl.lvs_clean is True


# ── Edge-branch coverage ─────────────────────────────────────────────


class TestASICFlowEdgeBranches:
    """Cover the non-open-source PDK manifest fallback, bundle serialisation,
    the unknown-PDK normaliser guard, the empty-corner slack sentinel, and the
    timing/metal-layer validation branches."""

    def test_file_manifest_for_non_open_source_pdk_omits_signoff_decks(self) -> None:
        """A commercial PDK (no bundled netgen/KLayout decks) resolves only the
        liberty/LEF paths; the setup/DRC/LVS deck fields stay empty."""
        pdk = PDKConfig(
            pdk_type=PDKType.TSMC28,
            liberty_file="lib/tsmc28.lib",
            lef_file="lef/tsmc28.lef",
            tech_lef="lef/tsmc28.tech.lef",
        )
        files = OpenSourcePDKResolver._file_manifest(pdk)
        assert files.liberty_file == "lib/tsmc28.lib"
        assert files.lef_file == "lef/tsmc28.lef"
        assert files.tech_lef == "lef/tsmc28.tech.lef"
        assert files.setup_tcl == ""
        assert files.drc_deck == ""
        assert files.lvs_setup == ""

    def test_bundle_to_dict_round_trips_paths(self, tmp_path: Path) -> None:
        """ASICFlowBundle.to_dict mirrors the bundle's own path fields."""
        bundle = generate_asic_flow_bundle(tmp_path, pdk_type="sky130")
        assert isinstance(bundle, ASICFlowBundle)
        payload = bundle.to_dict()
        assert payload["output_dir"] == bundle.output_dir
        assert payload["manifest_path"] == bundle.manifest_path
        assert payload["file_paths"] == dict(bundle.file_paths)

    def test_normalise_pdk_type_rejects_unknown_string(self) -> None:
        """An unrecognised PDK name is rejected with the list of valid types."""
        with pytest.raises(ValueError, match="unknown PDK type"):
            _normalise_pdk_type("not-a-real-pdk")

    def test_worst_slack_returns_sentinel_for_empty_corners(self) -> None:
        """With no corner data there is no worst slack, so a neutral sentinel is
        returned rather than raising on an empty min()."""
        assert MultiCornerAnalysis.worst_slack({}) == ("none", 0.0)

    def test_validate_pdk_flags_bad_clock_and_low_metal_layers(self) -> None:
        """A non-positive clock period is an error and fewer than three metal
        layers is a routing warning; the custom type skips the file checks."""
        pdk = PDKConfig(
            pdk_type=PDKType.CUSTOM,
            clock_period_ns=0.0,
            metal_layers=2,
            voltage_v=1.8,
        )
        result = validate_pdk(pdk)
        assert not result.valid
        assert any("clock_period_ns must be positive" in err for err in result.errors)
        assert any("metal layers" in warn for warn in result.warnings)
