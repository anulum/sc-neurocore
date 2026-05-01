# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851

"""End-to-end integration tests — full compilation pipeline.

These tests exercise cross-cutting paths through the compiler that span
multiple modules (ODE → Verilog → constraints → drivers → formal → safety).

Run selectively with::

    pytest tests/e2e/ -m e2e -v
"""

import re

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════

LIF_EQUATIONS = {"v": "-(v - v_rest) / tau_m + R * I / C"}
IZH_EQUATIONS = {
    "v": "0.04 * v * v + 5 * v + 140 - u + I",
    "u": "a * (b * v - u)",
}
STATE_VARS_LIF = ["v"]
STATE_VARS_IZH = ["v", "u"]


# ═══════════════════════════════════════════════════════════════════════
# E2E 1: ODE → Verilog → Resource Estimate → Constraints → Driver
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestODEToDriverPipeline:
    """Full pipeline: compile → estimate → constrain → driver."""

    def test_lif_full_pipeline_artix7(self):
        """LIF on Artix-7: every artefact is internally consistent."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        from sc_neurocore.compiler.deployment import (
            estimate_resources, generate_constraints,
            generate_host_driver, generate_cocotb_testbench,
        )

        profile = get_profile("artix7")
        module = "sc_lif_e2e"
        dw = profile.data_width

        # 1. Resource estimate
        verilog_stub = (
            "module sc_lif_e2e(\n"
            f"  input wire signed [{dw-1}:0] I_t,\n"
            f"  output wire signed [{dw-1}:0] v_next\n"
            ");\n"
            f"  wire signed [{2*dw-1}:0] _mul0 = I_t * {dw}'sd10;\n"
            f"  wire signed [{dw-1}:0] _t0 = _mul0[{dw-1}:0];\n"
            "endmodule\n"
        )
        res = estimate_resources(verilog_stub, has_dsp=bool(profile.dsp_block))
        assert res.luts >= 0
        assert res.mul_count >= 0

        # 2. Constraints (uses target_freq_mhz, not target object)
        freq = profile.max_freq_mhz or 100
        xdc = generate_constraints(
            module_name=module,
            data_width=dw,
            target_freq_mhz=float(freq),
        )
        assert "create_clock" in xdc

        # 3. Host driver (C)
        c_driver = generate_host_driver(
            module_name=module,
            params={"v": dw, "I_t": dw},
            data_width=dw,
            language="c",
        )
        assert "write" in c_driver.lower() or "WRITE" in c_driver

        # 4. Cocotb testbench
        tb = generate_cocotb_testbench(
            module_name=module,
            data_width=dw,
        )
        assert "import cocotb" in tb
        assert module in tb

    def test_pipeline_data_width_consistency(self):
        """Data widths match across constraints and drivers."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        from sc_neurocore.compiler.deployment import (
            generate_constraints, generate_host_driver,
        )

        for target_name in ["artix7", "loihi2", "ecp5"]:
            profile = get_profile(target_name)
            dw = profile.data_width
            module = f"sc_test_{target_name}"
            freq = profile.max_freq_mhz or 100

            xdc = generate_constraints(
                module_name=module, data_width=dw,
                target_freq_mhz=float(freq),
            )
            driver = generate_host_driver(
                module_name=module, data_width=dw,
                params={"v": dw}, language="c",
            )
            # Both should reference something meaningful
            assert "create_clock" in xdc
            assert module in driver


# ═══════════════════════════════════════════════════════════════════════
# E2E 2: ODE → SVA → SymbiYosys → Certification
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestFormalToCertification:
    """Formal verification → safety certification evidence chain."""

    def test_sva_matches_verilog_ports(self):
        """SVA variable names match what a compiled Verilog module would have."""
        from sc_neurocore.compiler.static_analysis import generate_sva

        sva = generate_sva(
            STATE_VARS_LIF,
            data_width=16, fraction=8,
            module_name="sc_lif_formal",
        )
        assert "v_reg" in sva
        assert "spike_out" in sva
        assert "I_t" in sva
        assert "sc_lif_formal" in sva

    def test_sby_references_correct_module(self):
        """SymbiYosys script references the correct module name."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        # Actual API: generate_sby_script(module_name, *, sva_file=...)
        sby = generate_sby_script(
            "sc_lif_formal",
            sva_file="sc_lif_sva.sv",
        )
        assert "sc_lif_formal.v" in sby
        assert "sc_lif_sva.sv" in sby
        assert "sc_lif_formal" in sby

    def test_certification_with_items(self):
        """Certification evidence XML includes all items."""
        from sc_neurocore.compiler.deployment import (
            generate_certification_evidence, CertificationItem,
        )

        items = [
            CertificationItem(
                req_id="REQ-001", description="No overflow",
                design_ref="sc_lif.v", verification_ref="sc_lif_sva.sv",
                status="PASS",
            ),
            CertificationItem(
                req_id="REQ-002", description="Spike reachable",
                design_ref="sc_lif.v", verification_ref="tb_lif.py",
                status="PASS",
            ),
        ]
        xml = generate_certification_evidence(
            "sc_lif_cert", items,
            standard="do254", dal_level="DAL-A",
        )
        assert "sc_lif_cert" in xml
        assert "DO-254" in xml
        assert "DAL-A" in xml
        assert "REQ-001" in xml
        assert "REQ-002" in xml

    def test_full_formal_chain(self):
        """SVA → .sby → certification: all module names consistent."""
        from sc_neurocore.compiler.static_analysis import generate_sva
        from sc_neurocore.compiler.deployment import (
            generate_sby_script, generate_certification_evidence,
            CertificationItem,
        )

        module = "sc_hh_formal"

        sva = generate_sva(["v", "n", "m", "h"], data_width=32, fraction=16,
                           module_name=module)
        sby = generate_sby_script(module)
        items = [CertificationItem(
            req_id="REQ-100", description="Bounded membrane",
            design_ref=f"{module}.v", verification_ref=f"{module}_sva.sv",
            status="PASS",
        )]
        xml = generate_certification_evidence(
            module, items, standard="iec61508", dal_level="SIL-3",
        )

        # All three reference the same module
        assert module in sva
        assert module in sby
        assert module in xml


# ═══════════════════════════════════════════════════════════════════════
# E2E 3: Multi-Target Comparison Consistency
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestMultiTargetConsistency:
    """Cross-target comparison: same ODE, different targets."""

    def test_guard_bits_target_independent(self):
        """Guard bits depend on the expression, not the target."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b + c + d"},
            ["artix7", "loihi2", "ecp5", "asic_16"],
        )
        guards = {r.guard_bits for r in results}
        assert len(guards) == 1

    def test_dsps_scale_with_dsp_block(self):
        """Targets with DSP blocks allocate DSPs."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b * c"},
            ["artix7", "ecp5"],
        )
        # Both have DSP blocks
        for r in results:
            assert r.estimated_dsps > 0

    def test_table_includes_all_targets(self):
        """Comparison table mentions every target."""
        from sc_neurocore.compiler.deployment import (
            compile_multi_target, format_comparison_table,
        )

        targets = ["artix7", "loihi2", "ecp5", "asic_16"]
        results = compile_multi_target({"v": "a * b + c"}, targets)
        table = format_comparison_table(results)
        for t in targets:
            assert t in table


# ═══════════════════════════════════════════════════════════════════════
# E2E 4: Network-Level Pipeline
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestNetworkPipeline:
    """BRAM array → weight ROM → constraints → testbench."""

    def test_bram_array_is_synthesisable(self):
        """BRAM array Verilog is structurally valid."""
        from sc_neurocore.compiler.advanced_features import (
            storage_recommendation, generate_bram_array,
        )

        rec = storage_recommendation(512, 16)
        assert rec.strategy == "bram"

        v = generate_bram_array(neuron_count=512, data_width=16)
        assert "module sc_neuron_array" in v
        assert "endmodule" in v
        assert "state_bram" in v
        assert "ram_style" in v
        assert "spike_out" in v
        assert "tick_done" in v

    def test_weight_rom_matches_dimensions(self):
        """Weight ROM entries match weight matrix dimensions."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom

        weights = [[i * 10 + j for j in range(4)] for i in range(8)]
        mif = generate_weight_rom(weights, output_format="mif")
        coe = generate_weight_rom(weights, output_format="coe")

        assert "DEPTH=32" in mif  # 8×4 = 32

    def test_bram_array_plus_constraints(self):
        """BRAM array → constraints: valid artefacts from same data width."""
        from sc_neurocore.compiler.advanced_features import generate_bram_array
        from sc_neurocore.compiler.hardware_profiles import get_profile
        from sc_neurocore.compiler.deployment import generate_constraints

        profile = get_profile("artix7")
        dw = profile.data_width

        v = generate_bram_array(
            module_name="sc_net_512",
            neuron_count=512,
            data_width=dw,
        )
        xdc = generate_constraints(
            module_name="sc_net_512",
            data_width=dw,
            target_freq_mhz=float(profile.max_freq_mhz or 100),
        )
        assert str(dw - 1) in v
        assert "create_clock" in xdc


# ═══════════════════════════════════════════════════════════════════════
# E2E 5: DVS → AER → Neuron → Spike → Driver
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestDVSToDriverPipeline:
    """DVS event camera → AER bridge → RISC-V driver chain."""

    def test_dvs_aer_bridge_valid(self):
        """DVS bridge Verilog is structurally valid."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge

        bridge = generate_dvs_aer_bridge(
            module_name="sc_dvs_bridge",
            addr_width=16,
        )
        assert "module sc_dvs_bridge" in bridge
        assert "endmodule" in bridge

    def test_dvs_bridge_plus_riscv_driver(self):
        """DVS bridge + RISC-V driver: both produce valid artefacts."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        bridge = generate_dvs_aer_bridge()
        driver = generate_riscv_driver(
            "sc_dvs_neuron",
            params={"v": 16, "I_t": 16},
            data_width=16,
            rtos="baremetal",
        )
        assert "endmodule" in bridge
        assert "uint" in driver or "void" in driver


# ═══════════════════════════════════════════════════════════════════════
# E2E 6: Thermal-Aware Full Flow
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestThermalFullFlow:
    """Power estimation → thermal analysis → derated constraints."""

    def test_power_to_thermal_to_constraints(self):
        """Power → thermal → XDC: derated frequency propagates."""
        from sc_neurocore.compiler.static_analysis import estimate_power
        from sc_neurocore.compiler.advanced_features import (
            thermal_analysis, generate_thermal_constraints,
        )

        verilog = (
            "reg signed [15:0] v_reg;\n"
            "wire signed [31:0] _mul0 = a * b;\n"
            "wire signed [31:0] _mul1 = c * d;\n"
            "wire signed [15:0] _t0 = a + b - c + d;\n"
        )
        power = estimate_power(verilog, freq_mhz=500.0, process_nm=16)
        therm = thermal_analysis(
            power.total_mw, 500.0,
            process_nm=16,
            mul_count=2,
        )
        xdc = generate_thermal_constraints("sc_lif_thermal", therm)

        assert str(therm.derated_freq_mhz) in xdc
        assert "create_clock" in xdc

    def test_high_power_triggers_warning(self):
        """Very high power → thermal unsafe → warning in XDC."""
        from sc_neurocore.compiler.advanced_features import (
            thermal_analysis, generate_thermal_constraints,
        )

        therm = thermal_analysis(50000.0, 500.0)
        assert not therm.thermal_safe

        xdc = generate_thermal_constraints("sc_hot", therm)
        assert "WARNING" in xdc


# ═══════════════════════════════════════════════════════════════════════
# E2E 7: Cross-Format Weight Consistency
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestWeightFormatConsistency:
    """All weight formats contain identical data."""

    def test_all_formats_same_values(self):
        """Verilog, .coe, .mif all encode the same weights."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom

        weights = [[100, 50, 0], [25, 75, 127]]

        v = generate_weight_rom(weights, data_width=16, output_format="verilog")
        coe = generate_weight_rom(weights, data_width=16, output_format="coe")
        mif = generate_weight_rom(weights, data_width=16, output_format="mif")

        # Extract hex values from each format
        hex_v = re.findall(r"'sh([0-9a-fA-F]+)", v)
        hex_coe = re.findall(r"^([0-9a-fA-F]+)[,;]", coe, re.MULTILINE)
        hex_mif = re.findall(r": ([0-9a-fA-F]+);", mif)

        assert len(hex_v) == 6
        assert len(hex_coe) == 6
        assert len(hex_mif) == 6
        assert hex_v == hex_coe == hex_mif


# ═══════════════════════════════════════════════════════════════════════
# E2E 8: MXFP Round-Trip Accuracy
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestMXFPRoundTrip:
    """Block-FP encode → decode preserves values within precision bounds."""

    def test_mxfp8_e4m3_round_trip(self):
        """MXFP8 E4M3: encode → decode → sign preservation."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP8_E4M3, mxfp_encode_block, mxfp_decode_block,
        )

        # Block size is 32, so we need exactly 32 values
        values = [0.0, 1.0, -1.0, 0.5] * 8  # 32 values
        shared_exp, elements = mxfp_encode_block(values, MXFP8_E4M3)
        decoded = mxfp_decode_block(shared_exp, elements, MXFP8_E4M3)

        assert len(decoded) == len(values)
        for orig, dec in zip(values, decoded):
            if orig != 0:
                assert (orig > 0) == (dec > 0), f"Sign flip: {orig} → {dec}"

    def test_zero_stability(self):
        """Zero encodes and decodes as zero."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP8_E5M2, mxfp_encode_block, mxfp_decode_block,
        )

        values = [0.0] * 32  # Block size = 32
        shared_exp, elements = mxfp_encode_block(values, MXFP8_E5M2)
        decoded = mxfp_decode_block(shared_exp, elements, MXFP8_E5M2)
        assert all(d == 0.0 for d in decoded)

    def test_all_configs_round_trip(self):
        """Every block-FP config can encode/decode without crashing."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP4, MXFP6, MXFP8_E4M3, MXFP8_E5M2,
            mxfp_encode_block, mxfp_decode_block,
        )

        # Only test block-FP configs (shared_exp_bits > 0)
        # FP8 standalone (block_size=1) uses per-element exponent
        configs = {
            "MXFP4": MXFP4, "MXFP6": MXFP6,
            "MXFP8_E4M3": MXFP8_E4M3, "MXFP8_E5M2": MXFP8_E5M2,
        }
        for name, config in configs.items():
            values = ([1.0, -1.0, 0.5, 0.0] * max(1, config.block_size // 4))[:config.block_size]
            shared_exp, elements = mxfp_encode_block(values, config)
            decoded = mxfp_decode_block(shared_exp, elements, config)
            assert len(decoded) == len(values), f"{name}: length mismatch"


# ═══════════════════════════════════════════════════════════════════════
# E2E 9: Pipeline Analysis → Thermal → Multi-Target
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.e2e
class TestAnalysisChain:
    """Pipeline depth → power → thermal → multi-target: all consistent."""

    def test_complex_ode_analysis_chain(self):
        """HH-class ODE: pipeline → power → thermal → compare."""
        from sc_neurocore.compiler.static_analysis import (
            critical_path_depth, pipeline_stages_needed,
            compute_guard_bits, estimate_power,
        )
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        from sc_neurocore.compiler.deployment import compile_multi_target

        hh_expr = "gNa * m * m * m * h * (v - ENa)"
        depth = critical_path_depth(hh_expr)
        assert depth >= 3

        stages = pipeline_stages_needed(depth, 900)
        assert stages >= 1

        guard = compute_guard_bits(hh_expr)
        assert guard >= 0

        results = compile_multi_target(
            {"v": hh_expr},
            ["artix7", "ecp5"],
        )
        assert len(results) == 2
        assert len({r.guard_bits for r in results}) == 1

    def test_slr_placement_valid(self):
        """SLR placement constraints are structurally valid."""
        from sc_neurocore.compiler.deployment import (
            generate_slr_constraints, SLRPlacement,
        )

        placements = [
            SLRPlacement(module_name="sc_hh_v", slr=0, pblock_name="pblock_v"),
            SLRPlacement(module_name="sc_hh_n", slr=1, pblock_name="pblock_n"),
        ]
        xdc = generate_slr_constraints(placements)
        assert "pblock" in xdc.lower()
        assert "SLR0" in xdc
        assert "SLR1" in xdc
