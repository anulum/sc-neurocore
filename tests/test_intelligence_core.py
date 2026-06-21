# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import sc_neurocore
from importlib.metadata import PackageNotFoundError, version

import pytest
import json
import numpy as np
import sc_neurocore_engine as v3


def _assert_engine_version_matches_core() -> None:
    """Validate source-tree and installed-wheel version surfaces."""

    assert v3.__version__ == sc_neurocore.__version__
    try:
        installed_version = version("sc-neurocore-engine")
    except PackageNotFoundError:
        return
    assert installed_version == sc_neurocore.__version__


class TestQuantisationSweep:
    """Quantisation design-space exploration."""

    def test_default_sweep(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b + c"})
        assert len(results) == 7  # [4, 8, 12, 16, 20, 24, 32]

    def test_widths_sorted(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b"})
        widths = [r.data_width for r in results]
        assert widths == sorted(widths)

    def test_luts_grow_with_width(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a + b + c"})
        luts = [r.estimated_luts for r in results]
        assert luts == sorted(luts)  # Monotonically increasing

    def test_precision_improves_with_width(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep({"v": "a * b"})
        steps = [r.min_step for r in results]
        assert steps == sorted(steps, reverse=True)  # Smaller step = better

    def test_custom_widths(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {"v": "a + b"},
            widths=[8, 16, 32],
        )
        assert len(results) == 3

    def test_format_report(self):
        from sc_neurocore.compiler.intelligence import (
            auto_quantisation_sweep,
            format_quantisation_report,
        )

        results = auto_quantisation_sweep({"v": "a * b + c"})
        report = format_quantisation_report(results)
        assert "Q-format" in report
        assert "LUTs" in report
        assert "LSB Step" in report

    def test_target_affects_dsps(self):
        """Targets with DSP blocks should show DSP usage."""
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        r_artix = auto_quantisation_sweep(
            {"v": "a * b"},
            target="artix7",
        )
        r_ice40 = auto_quantisation_sweep(
            {"v": "a * b"},
            target="bae_rad750",
        )
        # Artix has DSP48E1, RAD750 doesn't
        assert all(r.estimated_dsps > 0 for r in r_artix)
        assert all(r.estimated_dsps == 0 for r in r_ice40)

    def test_izh_multi_equation(self):
        from sc_neurocore.compiler.intelligence import auto_quantisation_sweep

        results = auto_quantisation_sweep(
            {
                "v": "0.04 * v * v + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            }
        )
        # More equations → more FFs
        for r in results:
            assert r.estimated_ffs == 2 * r.data_width  # 2 state vars


class TestHLSCppExport:
    """Vitis/Catapult HLS C++ translation."""

    def test_vitis_export(self):
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_lif",
            {"v": "v + I_t - v * leak"},
            data_width=16,
            fraction=8,
        )
        assert "ap_fixed<16,8>" in cpp
        assert "#pragma HLS PIPELINE" in cpp
        assert "void sc_lif(" in cpp
        assert "V_THRESH" in cpp

    def test_catapult_export(self):
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp(
            "sc_hh",
            {"v": "a + b", "n": "c * d"},
            hls_tool="catapult",
        )
        assert "Catapult" in cpp
        assert "v_next" in cpp
        assert "n_next" in cpp

    def test_include_guard(self):
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "a + b"})
        assert "#ifndef SC_LIF_HLS_H" in cpp
        assert "#endif" in cpp

    def test_state_struct(self):
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_izh", {"v": "a", "u": "b"})
        assert "struct sc_izh_state" in cpp
        assert "fp_t v;" in cpp
        assert "fp_t u;" in cpp

    def test_spike_detection(self):
        from sc_neurocore.compiler.intelligence import generate_hls_cpp

        cpp = generate_hls_cpp("sc_lif", {"v": "v + I_t"})
        assert "spike_out" in cpp
        assert "V_THRESH" in cpp


class TestOnChipLearning:
    """STDP / reward-modulated learning parameter export."""

    def test_default_stdp_params(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params()
        assert p.learning_rule == "stdp"
        assert p.tau_plus_ms == 20.0
        assert p.a_plus == 0.01
        assert p.target_platform == "akida2"

    def test_rstdp_params(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params(
            learning_rule="rstdp",
            reward_tau_ms=500.0,
        )
        assert p.learning_rule == "rstdp"
        assert p.reward_tau_ms == 500.0

    def test_json_export(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params()
        cfg = export_learning_config(p, output_format="json")
        data = json.loads(cfg)
        assert data["learning_rule"] == "stdp"
        assert "time_constants" in data
        assert "weight_bounds" in data

    def test_yaml_export(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params(target="brainscales2")
        cfg = export_learning_config(p, output_format="yaml")
        assert "learning_rule: stdp" in cfg
        assert "brainscales2" in cfg
        assert "tau_plus_ms:" in cfg

    def test_rejects_unknown_export_format(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        params = generate_learning_params()
        with pytest.raises(ValueError, match="Unsupported learning config format"):
            export_learning_config(params, output_format="toml")

    def test_custom_weight_bounds(self):
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params(w_max=2.0, w_min=-1.0)
        assert p.w_max == 2.0
        assert p.w_min == -1.0


class TestSymbiYosys:
    """Tests for SymbiYosys .sby script generation."""

    def test_basic_bmc_script(self):
        """Default BMC script has required sections."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif")
        assert "[options]" in sby
        assert "mode bmc" in sby
        assert "depth 20" in sby
        assert "[engines]" in sby
        assert "smtbmc boolector" in sby
        assert "[script]" in sby
        assert "read_verilog -formal sc_lif.v" in sby
        assert "read_verilog -sv -formal sc_lif_sva.sv" in sby
        assert "prep -top sc_lif" in sby
        assert "[files]" in sby

    def test_prove_mode(self):
        """Prove mode sets induction."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", mode="prove", depth=50)
        assert "mode prove" in sby
        assert "depth 50" in sby

    def test_cover_mode(self):
        """Cover mode for reachability."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", mode="cover")
        assert "mode cover" in sby

    def test_custom_sva_file(self):
        """Custom SVA file path."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", sva_file="my_props.sv")
        assert "my_props.sv" in sby

    def test_z3_engine(self):
        """Z3 solver engine."""
        from sc_neurocore.compiler.deployment import generate_sby_script

        sby = generate_sby_script("sc_lif", engine="z3")
        assert "smtbmc z3" in sby


class TestRISCVDriver:
    """Tests for RISC-V C driver generation."""

    def test_baremetal_driver(self):
        """Baremetal driver has MMIO macros and functions."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16})
        assert "#ifndef SC_LIF_RISCV_H" in c
        assert "MMIO_WR" in c
        assert "MMIO_RD" in c
        assert "sc_lif_enable" in c
        assert "sc_lif_reset" in c
        assert "sc_lif_set_current" in c
        assert "sc_lif_read_current" in c
        assert "sc_lif_get_spikes" in c
        assert "sc_lif_encode" in c
        assert "sc_lif_set_tau" in c
        assert "sc_lif_set_vth" in c
        assert "volatile" in c

    def test_freertos_template(self):
        """FreeRTOS template includes task and timer."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="freertos")
        assert "FreeRTOS.h" in c
        assert "xTaskCreate" in c
        assert "sc_lif_tick" in c
        assert "sc_lif_start_rtos" in c
        assert "vTaskDelay" in c
        assert "float I = sc_lif_read_current();" in c
        assert "TODO" not in c

    def test_zephyr_template(self):
        """Zephyr template includes thread and K_THREAD_DEFINE."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="zephyr")
        assert "zephyr/kernel.h" in c
        assert "K_THREAD_DEFINE" in c
        assert "k_msleep" in c
        assert "sc_lif_set_current(sc_lif_read_current())" in c

    def test_custom_base_address(self):
        """Custom base address propagates."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {}, base_address=0x8000_0000)
        assert "0x80000000" in c

    def test_param_registers(self):
        """Per-parameter register definitions."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16, "leak": 16})
        assert "SC_LIF_TAU" in c
        assert "SC_LIF_VTH" in c
        assert "SC_LIF_LEAK" in c


class TestDVSBridge:
    """Tests for DVS→AER bridge Verilog generation."""

    def test_basic_bridge(self):
        """Default bridge generates valid Verilog."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge()
        assert "module sc_dvs_aer_bridge" in v
        assert "dvs_valid" in v
        assert "dvs_ready" in v
        assert "aer_req" in v
        assert "aer_ack" in v
        assert "fifo_mem" in v
        assert "endmodule" in v

    def test_custom_widths(self):
        """Custom address and timestamp widths."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(addr_width=20, timestamp_width=48)
        assert "[19:0]" in v
        assert "[47:0]" in v

    def test_custom_module_name(self):
        """Custom module name."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(module_name="my_dvs_bridge")
        assert "module my_dvs_bridge" in v

    def test_fifo_depth(self):
        """FIFO depth affects address widths."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(fifo_depth=128)
        assert "[0:127]" in v  # 128-deep FIFO

    def test_polarity_bit_included(self):
        """Polarity bit appears in ports."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge(polarity_bit=True)
        assert "dvs_polarity" in v
        assert "aer_polarity" in v

    def test_overflow_flag(self):
        """FIFO overflow detection present."""
        from sc_neurocore.compiler.intelligence import generate_dvs_aer_bridge

        v = generate_dvs_aer_bridge()
        assert "fifo_overflow" in v
        assert "overflow_r" in v


class TestOpenSourceMakefile:
    """Open-source FPGA build recipe generation."""

    def test_ice40_recipe_uses_icepack_flow(self):
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        makefile = generate_oss_makefile("sc_lif", target="ice40")
        assert "nextpnr-ice40" in makefile
        assert "icepack" in makefile

    def test_ecp5_recipe_uses_ecppack_flow(self):
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        makefile = generate_oss_makefile("sc_lif", target="ecp5", device="um5g-85k")
        assert "nextpnr-ecp5 --um5g-85k" in makefile
        assert "ecppack" in makefile

    def test_rejects_unknown_open_source_target(self):
        from sc_neurocore.compiler.intelligence import generate_oss_makefile

        with pytest.raises(ValueError, match="Unsupported open-source FPGA target"):
            generate_oss_makefile("sc_lif", target="gowin")  # type: ignore[arg-type]


class TestSLRPlacement:
    """Tests for multi-die SLR constraint generation."""

    def test_single_slr(self):
        """Single SLR placement generates PBLOCK."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("neuron_array", slr=0),
            ]
        )
        assert "create_pblock pblock_slr0" in xdc
        assert "SLR0" in xdc
        # No inter-SLR directives for single SLR
        assert "REGISTER_DUPLICATION" not in xdc

    def test_multi_slr_pipeline_regs(self):
        """Multi-SLR adds pipeline register directives."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("input_stage", slr=0),
                SLRPlacement("compute_stage", slr=1),
            ]
        )
        assert "SLR0" in xdc
        assert "SLR1" in xdc
        assert "REGISTER_DUPLICATION" in xdc
        assert "set_max_delay" in xdc

    def test_no_pipeline_regs(self):
        """Opt-out of pipeline register insertion."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [SLRPlacement("a", 0), SLRPlacement("b", 1)],
            insert_pipeline_regs=False,
        )
        assert "REGISTER_DUPLICATION" not in xdc

    def test_custom_pblock_name(self):
        """Custom PBLOCK name."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement,
            generate_slr_constraints,
        )

        xdc = generate_slr_constraints(
            [
                SLRPlacement("core", slr=2, pblock_name="pblock_core"),
            ]
        )
        assert "create_pblock pblock_core" in xdc

    def test_auto_pblock_name(self):
        """Auto-generated PBLOCK name from SLR index."""
        from sc_neurocore.compiler.deployment import SLRPlacement

        p = SLRPlacement("test", slr=3)
        assert p.pblock_name == "pblock_slr3"


class TestMXFP:
    """Tests for MXFP / Block-FP encoding/decoding."""

    def test_mxfp4_config(self):
        """MXFP4 config matches OCP spec."""
        from sc_neurocore.compiler.intelligence import MXFP4

        assert MXFP4.element_bits == 4
        assert MXFP4.block_size == 32
        assert MXFP4.shared_exp_bits == 8
        assert MXFP4.label == "MXFP4"
        assert MXFP4.bits_per_block == 8 + 32 * 4  # 136

    def test_mxfp8_e4m3_config(self):
        """MXFP8 E4M3 config."""
        from sc_neurocore.compiler.intelligence import MXFP8_E4M3

        assert MXFP8_E4M3.element_bits == 8
        assert MXFP8_E4M3.exp_bits == 4
        assert MXFP8_E4M3.mantissa_bits == 3

    def test_fp8_no_shared_exp(self):
        """IEEE FP8 has no shared exponent (block_size=1)."""
        from sc_neurocore.compiler.intelligence import FP8_E4M3

        assert FP8_E4M3.block_size == 1
        assert FP8_E4M3.shared_exp_bits == 0

    def test_encode_decode_roundtrip_mxfp4(self):
        """MXFP4 encode→decode roundtrip preserves sign and order."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        values = [float(i) / 32 for i in range(32)]
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        # Order preserved
        for i in range(1, len(decoded)):
            assert decoded[i] >= decoded[i - 1]

    def test_encode_all_zeros(self):
        """All-zero block returns zero exponent."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
        )

        exp, encoded = mxfp_encode_block([0.0] * 32, MXFP4)
        assert exp == 0
        assert all(e == 0 for e in encoded)

    def test_block_size_mismatch_raises(self):
        """Wrong block size raises ValueError."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
        )

        with pytest.raises(ValueError, match="Block size"):
            mxfp_encode_block([1.0, 2.0], MXFP4)

    def test_negative_values(self):
        """Negative values have sign bit set."""
        from sc_neurocore.compiler.intelligence import (
            MXFP4,
            mxfp_encode_block,
            mxfp_decode_block,
        )

        values = [-1.0] * 32
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        assert all(d < 0 for d in decoded)

    def test_mxfp6_exists(self):
        """MXFP6 config exists."""
        from sc_neurocore.compiler.intelligence import MXFP6

        assert MXFP6.element_bits == 6


class TestCertificationEvidence:
    """Tests for safety-critical certification evidence generation."""

    def test_do254_xml(self):
        """DO-254 evidence generates valid XML structure."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        items = [
            CertificationItem("REQ-001", "No overflow", "sc_lif.v", "sc_lif_sva.sv", "PASS"),
            CertificationItem("REQ-002", "Reset clears state", "sc_lif.v", "test_reset", "PASS"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert '<?xml version="1.0"' in xml
        assert "<certification_evidence>" in xml
        assert "<module>sc_lif</module>" in xml
        assert "RTCA DO-254" in xml
        assert "DAL-C" in xml
        assert 'passed="2"' in xml
        assert 'coverage="100.0"' in xml
        assert 'id="REQ-001"' in xml

    def test_iec61508_standard(self):
        """IEC 61508 standard label."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        xml = generate_certification_evidence(
            "sc_lif",
            [CertificationItem("R1", "test", "d", "v", "PASS")],
            standard="iec61508",
            dal_level="SIL-3",
        )
        assert "IEC 61508" in xml
        assert "SIL-3" in xml

    def test_iso26262_standard(self):
        """ISO 26262 standard label."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        xml = generate_certification_evidence(
            "sc_lif",
            [CertificationItem("R1", "test", "d", "v", "FAIL")],
            standard="iso26262",
            dal_level="ASIL-D",
        )
        assert "ISO 26262" in xml
        assert "ASIL-D" in xml
        assert 'failed="1"' in xml

    def test_mixed_status_coverage(self):
        """Coverage calculation with mixed statuses."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        items = [
            CertificationItem("R1", "a", "d", "v", "PASS"),
            CertificationItem("R2", "b", "d", "v", "FAIL"),
            CertificationItem("R3", "c", "d", "v", "UNTESTED"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert 'total="3"' in xml
        assert 'passed="1"' in xml
        assert 'failed="1"' in xml
        assert 'coverage="33.3"' in xml

    def test_empty_items(self):
        """Empty items list produces valid XML with 0% coverage."""
        from sc_neurocore.compiler.deployment import generate_certification_evidence

        xml = generate_certification_evidence("sc_lif", [])
        assert 'total="0"' in xml
        assert 'coverage="0.0"' in xml


class TestPipelineAnalysis:
    """Tests for critical path depth and pipeline budget."""

    def test_no_multiply(self):
        """Pure addition has zero depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a + b + c") == 0

    def test_single_multiply(self):
        """Single multiply has depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b") == 1

    def test_chained_multiply(self):
        """Chained a * b * c has depth 2."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c") == 2

    def test_deep_chain(self):
        """a * b * c * d has depth 3."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c * d") == 3

    def test_mixed(self):
        """a * b + c * d: both branches have depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b + c * d") == 1

    def test_divide_counts(self):
        """Division counts as multiplicative depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a / b") == 1

    def test_no_pipeline_needed_slow(self):
        """No pipeline at 100 MHz with depth 1."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(1, 100) == 0

    def test_pipeline_needed_fast(self):
        """Pipeline needed at 900 MHz with depth 4."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        stages = pipeline_stages_needed(4, 900)
        assert stages >= 1  # 4 × 3.0 ns = 12 ns > 1.11 ns period

    def test_pipeline_zero_depth(self):
        """Zero depth → zero stages."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(0, 900) == 0

    def test_pipeline_analysis_multi(self):
        """Multi-ODE pipeline analysis."""
        from sc_neurocore.compiler.static_analysis import pipeline_analysis

        result = pipeline_analysis(
            {"v": "a * b * c + d", "w": "e + f"},
            target_freq_mhz=500,
        )
        assert result["v"]["depth"] == 2
        assert result["w"]["depth"] == 0
        assert result["w"]["stages"] == 0
        assert "achievable_mhz" in result["v"]


class TestPowerEstimation:
    """Tests for compile-time power estimation."""

    def test_basic_power(self):
        """Basic LIF-like Verilog produces non-zero power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        verilog = """
        reg signed [15:0] v_reg;
        wire signed [31:0] _mul0 = a * b;
        wire signed [15:0] _t0 = a + b - c;
        """
        p = estimate_power(verilog)
        assert p.dynamic_mw >= 0
        assert p.static_mw >= 0
        assert p.total_mw > 0
        assert p.toggle_rate >= 0

    def test_higher_freq_more_power(self):
        """Higher frequency = more dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p100 = estimate_power(v, freq_mhz=100)
        p500 = estimate_power(v, freq_mhz=500)
        assert p500.dynamic_mw > p100.dynamic_mw

    def test_energy_per_spike(self):
        """Energy per spike is computed from power and rate."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p = estimate_power(v, spike_rate_hz=100.0)
        assert p.energy_per_spike_nj > 0

    def test_different_process(self):
        """Smaller process = less capacitance = less dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p28 = estimate_power(v, process_nm=28)
        p7 = estimate_power(v, process_nm=7)
        assert p7.dynamic_mw < p28.dynamic_mw

    def test_empty_verilog(self):
        """Empty Verilog produces near-zero power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        p = estimate_power("")
        assert p.total_mw == 0

    def test_vcd_activity_drives_measured_toggle_rate(self):
        """VCD switching activity overrides structural default toggles."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        vcd = """
        $timescale 1ns $end
        $scope module top $end
        $var wire 4 ! data [3:0] $end
        $upscope $end
        $enddefinitions $end
        #0
        b0000 !
        #10
        b1111 !
        #20
        b1010 !
        """

        p = estimate_power("", activity_vcd=vcd, vcd_time_units_per_cycle=10)

        assert p.dynamic_mw > 0
        assert p.total_mw == p.dynamic_mw
        assert p.toggle_rate == 0.75

    def test_vcd_activity_rejects_invalid_cycle_scale(self):
        """VCD-derived activity needs a positive time-unit to cycle scale."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        with pytest.raises(ValueError, match="vcd_time_units_per_cycle"):
            estimate_power("", activity_vcd="$enddefinitions $end", vcd_time_units_per_cycle=0)


class TestMultiTarget:
    """Tests for multi-target --compare compilation."""

    def test_basic_multi_target(self):
        """Compile LIF to 3 targets and get results."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "-(v - v_rest) / tau + R * I"},
            ["artix7", "loihi2", "asic_16"],
        )
        assert len(results) == 3
        targets = [r.target for r in results]
        assert "artix7" in targets
        assert "loihi2" in targets
        assert "asic_16" in targets

    def test_data_widths_differ(self):
        """Different targets have different data widths."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        r_map = {r.target: r for r in results}
        assert r_map["artix7"].data_width != r_map["loihi2"].data_width

    def test_guard_bits_consistent(self):
        """Guard bits should be same for all targets (expression-dependent)."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b + c + d"},
            ["artix7", "ice40", "ecp5"],
        )
        guards = [r.guard_bits for r in results]
        assert len(set(guards)) == 1  # All same

    def test_format_comparison_table(self):
        """Table formatter produces markdown."""
        from sc_neurocore.compiler.deployment import (
            compile_multi_target,
            format_comparison_table,
        )

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "ice40"],
        )
        table = format_comparison_table(results)
        assert "| Target" in table
        assert "artix7" in table
        assert "ice40" in table

    def test_single_target(self):
        """Single target still works."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b"},
            ["artix7"],
        )
        assert len(results) == 1
        assert results[0].target == "artix7"

    def test_dsp_allocation(self):
        """Targets with DSP blocks allocate multipliers to DSPs."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b * c"},
            ["artix7"],  # has DSP48E1
        )
        assert results[0].estimated_dsps > 0


class TestStorageRecommendation:
    """Tests for BRAM/register storage strategy."""

    def test_small_uses_registers(self):
        """≤64 neurons → registers."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(32, 16)
        assert rec.strategy == "registers"
        assert rec.total_bits == 32 * 16

    def test_medium_uses_bram(self):
        """65–16K neurons → BRAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(1024, 16)
        assert rec.strategy == "bram"
        assert rec.total_bits == 1024 * 16

    def test_large_with_uram(self):
        """≥16K neurons with URAM → URAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(20000, 16, has_uram=True)
        assert rec.strategy == "uram"
        assert rec.uram_used >= 1

    def test_large_without_uram_uses_bram(self):
        """Large without URAM → falls back to BRAM."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(20000, 16, has_uram=False)
        assert rec.strategy == "bram"

    def test_custom_threshold(self):
        """Custom register threshold."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(100, 16, register_threshold=128)
        assert rec.strategy == "registers"

    def test_bram_18k_for_small(self):
        """Small BRAM uses 18Kb tile."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(128, 16)  # 2048 bits, fits in 18Kb
        assert rec.strategy == "bram"
        assert rec.bram_18k_used == 1
        assert rec.bram_36k_used == 0

    def test_bram_36k_for_large(self):
        """Larger BRAM uses 36Kb tiles."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(4096, 16)  # 65536 bits
        assert rec.strategy == "bram"
        assert rec.bram_36k_used >= 1

    def test_reason_populated(self):
        """Reason string is non-empty."""
        from sc_neurocore.compiler.intelligence import storage_recommendation

        rec = storage_recommendation(10, 16)
        assert len(rec.reason) > 0


class TestWeightROM:
    """Tests for synaptic weight ROM generation."""

    def test_verilog_rom(self):
        """Verilog ROM module."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2], [3, 4]]
        v = generate_weight_rom(w)
        assert "module sc_weight_rom" in v
        assert "case" in v
        assert "endmodule" in v

    def test_coe_format(self):
        """Xilinx .coe format."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[10, 20], [30, 40]]
        coe = generate_weight_rom(w, output_format="coe")
        assert "memory_initialization_radix=16" in coe
        assert "memory_initialization_vector=" in coe

    def test_mif_format(self):
        """Intel .mif format."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[10, 20], [30, 40]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "WIDTH=16" in mif
        assert "DEPTH=4" in mif
        assert "CONTENT BEGIN" in mif
        assert "END;" in mif

    def test_custom_module_name(self):
        """Custom ROM module name."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2]]
        v = generate_weight_rom(w, module_name="my_weights")
        assert "module my_weights" in v

    def test_correct_entry_count(self):
        """Correct number of entries in ROM."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1, 2, 3], [4, 5, 6]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "DEPTH=6" in mif

    def test_data_width_propagates(self):
        """Custom data width propagates."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        w = [[1]]
        mif = generate_weight_rom(w, data_width=8, output_format="mif")
        assert "WIDTH=8" in mif

    def test_rejects_unknown_output_format(self):
        """Unknown memory formats must not silently emit Verilog."""
        from sc_neurocore.compiler.intelligence import generate_weight_rom

        with pytest.raises(ValueError, match="Unsupported weight ROM format"):
            generate_weight_rom([[1]], output_format="hex")


class TestWeightNoise:
    """Analog device-variation weight-noise injection."""

    def test_gaussian_noise_is_seeded_and_shape_preserving(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        weights = [[0.25, -0.5], [1.0, 0.0]]
        first = inject_weight_noise(weights, noise_model="gaussian", sigma=0.1, seed=7)
        second = inject_weight_noise(weights, noise_model="gaussian", sigma=0.1, seed=7)

        assert first == second
        assert len(first) == len(weights)
        assert all(len(row) == len(src) for row, src in zip(first, weights, strict=True))

    def test_uniform_noise_is_bounded_by_sigma_scale(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        weights = [[-2.0, 0.0, 2.0]]
        noisy = inject_weight_noise(weights, noise_model="uniform", sigma=0.25, seed=11)

        for original, perturbed in zip(weights[0], noisy[0], strict=True):
            assert abs(perturbed - original) <= 0.5

    def test_zero_matrix_has_absolute_noise_scale_fallback(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        noisy = inject_weight_noise([[0.0, 0.0]], noise_model="uniform", sigma=0.1, seed=3)

        assert all(-0.1 <= value <= 0.1 for value in noisy[0])

    def test_rejects_unknown_noise_model(self):
        from sc_neurocore.compiler.intelligence import inject_weight_noise

        with pytest.raises(ValueError, match="Unsupported weight noise model"):
            inject_weight_noise([[1.0]], noise_model="triangular")


class TestTimescalePartitioner:
    """Multi-timescale ODE partitioning."""

    def test_single_timescale(self):
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales({"v": "a + b"})
        assert len(p.fast_equations) == 1
        assert len(p.slow_equations) == 0

    def test_explicit_separation(self):
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.fast_equations
        assert "w" in p.slow_equations
        assert p.slow_clock_div >= 2

    def test_cdc_signals(self):
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "v * c"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.cdc_signals

    def test_all_fast(self):
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
        )

        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 2.0},
        )
        assert len(p.slow_equations) == 0


class TestDriftCompensation:
    """Analog drift compensation controller."""

    def test_basic_compensator(self):
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator("sc_analog")
        assert "module sc_analog_drift_ctrl" in d.verilog_controller
        assert "endmodule" in d.verilog_controller
        assert d.refresh_interval_ms > 0
        assert d.compensation_method == "periodic_refresh"

    def test_fast_drift(self):
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator(
            "sc_rram",
            drift_rate_per_day=0.1,
            max_drift_tolerance=0.01,
        )
        # Should refresh very frequently
        assert d.refresh_interval_ms < 10_000_000

    def test_verilog_contains_counter(self):
        from sc_neurocore.compiler.intelligence import (
            generate_drift_compensator,
        )

        d = generate_drift_compensator("sc_mem")
        assert "counter" in d.verilog_controller
        assert "refresh_trigger" in d.verilog_controller
        assert "REFRESH_CYCLES" in d.verilog_controller


class TestHeterogeneousDispatch:
    """Multi-backend SNN dispatch."""

    def test_two_backends(self):
        from sc_neurocore.compiler.intelligence import (
            plan_heterogeneous_dispatch,
        )

        plan = plan_heterogeneous_dispatch(
            {"v": "a + b", "u": "c * d"},
            ["fpga", "gpu"],
        )
        assert "fpga" in plan.backends
        assert "gpu" in plan.backends
        assert plan.estimated_speedup > 1.0

    def test_single_backend(self):
        from sc_neurocore.compiler.intelligence import (
            plan_heterogeneous_dispatch,
        )

        plan = plan_heterogeneous_dispatch(
            {"v": "a + b"},
            ["fpga"],
        )
        assert len(plan.sync_barriers) == 0
        assert plan.total_neurons_per_backend["fpga"] == 1000

    def test_three_backends(self):
        from sc_neurocore.compiler.intelligence import (
            plan_heterogeneous_dispatch,
        )

        plan = plan_heterogeneous_dispatch(
            {"v": "a", "u": "b", "w": "c"},
            ["fpga", "mcu", "gpu"],
            neuron_count=3000,
        )
        assert len(plan.sync_barriers) == 2
        total = sum(plan.total_neurons_per_backend.values())
        assert total == 3000

    def test_neuron_distribution(self):
        from sc_neurocore.compiler.intelligence import (
            plan_heterogeneous_dispatch,
        )

        plan = plan_heterogeneous_dispatch(
            {"v": "a", "u": "b"},
            ["fpga", "gpu"],
            neuron_count=100,
        )
        total = sum(plan.total_neurons_per_backend.values())
        assert total == 100


class TestWave6Integration:
    """Cross-feature integration tests."""

    def test_provenance_then_compliance(self):
        """Provenance chain enables compliance coverage."""
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
            generate_compliance_matrix,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert len(chain) == 3
        matrix = generate_compliance_matrix(
            "sc_lif",
            has_provenance=True,
            has_tmr=True,
            has_checksum=True,
            has_sva=True,
        )
        all_covered = all(e.status == "covered" for e in matrix)
        assert all_covered

    def test_timescale_then_dispatch(self):
        """Partition timescales, then dispatch to backends."""
        from sc_neurocore.compiler.intelligence import (
            partition_timescales,
            plan_heterogeneous_dispatch,
        )

        part = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        all_eqs = {**part.fast_equations, **part.slow_equations}
        plan = plan_heterogeneous_dispatch(
            all_eqs,
            ["fpga", "mcu"],
        )
        assert plan.estimated_speedup > 1.0

    def test_equivalence_then_lint(self):
        """Generate proof sketch, then lint for side channels."""
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
            lint_side_channels,
        )

        sketch = generate_equivalence_sketch(
            "sc_hh",
            {"v": "a * b / c + d"},
        )
        findings = lint_side_channels({"v": "a * b / c + d"})
        assert sketch.quantisation_bound > 0
        assert len(findings) >= 3  # div + mul + spike

    def test_energy_schedule_for_mcu(self):
        """Energy schedule on edge MCU profile."""
        from sc_neurocore.compiler.platforms import get_profile
        from sc_neurocore.compiler.intelligence import (
            generate_energy_schedule,
        )

        p = get_profile("esp32_s3")
        assert p.platform_class == "edge_mcu"
        s = generate_energy_schedule(500, energy_budget_uj=5.0)
        assert s.neurons_per_epoch <= 500


class TestPartialReconfig:
    def test_basic_plan(self):
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration({"v": "a", "u": "b"})
        assert plan.total_regions == 2
        assert plan.bitstream_count == 2
        assert len(plan.schedule) == 2

    def test_single_var(self):
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration({"v": "a"})
        assert plan.total_regions == 1

    def test_custom_slots(self):
        from sc_neurocore.compiler.intelligence import (
            plan_partial_reconfiguration,
        )

        plan = plan_partial_reconfiguration(
            {"v": "a", "u": "b", "w": "c"},
            time_slots=4,
        )
        assert plan.bitstream_count == 4


class TestCompilationCache:
    def test_miss_then_hit(self):
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        assert cache.misses == 1
        cache.put(eqs, "artix7", 16, 8, {"verilog": "..."})
        result = cache.get(eqs, "artix7")
        assert result is not None
        assert cache.hits == 1

    def test_different_target_misses(self):
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        cache.put(eqs, "artix7", 16, 8, {"v": "data"})
        assert cache.get(eqs, "loihi2") is None

    def test_size(self):
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        cache.put({"v": "a"}, "artix7", 16, 8, {})
        cache.put({"v": "b"}, "artix7", 16, 8, {})
        assert cache.size == 2


class TestNetworkTopology:
    def test_basic_partition(self):
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.num_chips == 2
        assert len(plan.chip_assignment) == 5

    def test_all_intra(self):
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1], 1: [0]}
        plan = optimize_network_topology(adj, num_chips=1)
        assert plan.inter_chip_spikes == 0

    def test_bandwidth_reduction(self):
        from sc_neurocore.compiler.intelligence import (
            optimize_network_topology,
        )

        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.bandwidth_reduction >= 0.0


class TestNIRImport:
    def test_lif_import(self):
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph(
            {
                "nodes": {"n0": {"type": "LIF", "tau": 20}},
                "edges": [],
            }
        )
        assert "n0" in g.equations
        assert "20" in g.equations["n0"]

    def test_edges(self):
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph(
            {
                "nodes": {"a": {"type": "LIF"}, "b": {"type": "LIF"}},
                "edges": [["a", "b"]],
            }
        )
        assert ("a", "b") in g.edges

    def test_framework(self):
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {}, "edges": []}, framework="Norse")
        assert g.framework == "Norse"

    def test_izhikevich_import(self):
        """An Izhikevich node maps to its quadratic membrane equation."""
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {"n0": {"type": "Izhikevich"}}, "edges": []})
        assert "0.04" in g.equations["n0"]

    def test_unknown_type_falls_back_to_leaky_equation(self):
        """An unrecognised node type falls back to a generic leaky equation."""
        from sc_neurocore.compiler.intelligence import import_nir_graph

        g = import_nir_graph({"nodes": {"n0": {"type": "Mystery", "tau": 5.0}}, "edges": []})
        assert "5.0" in g.equations["n0"]


class TestDebugProbes:
    def test_xilinx(self):
        from sc_neurocore.compiler.intelligence import insert_debug_probes

        p = insert_debug_probes("sc_lif", {"v": "a"})
        assert p.probe_type == "ila"
        assert "v" in p.signals
        assert "create_debug_core" in p.tcl_commands

    def test_intel(self):
        from sc_neurocore.compiler.intelligence import insert_debug_probes

        p = insert_debug_probes("sc_lif", {"v": "a"}, vendor="intel")
        assert p.probe_type == "signaltap"


class TestReliability:
    def test_nominal(self):
        from sc_neurocore.compiler.intelligence import predict_reliability

        r = predict_reliability(voltage_v=0.9, temperature_c=25.0)
        assert r.mttf_years > 0
        assert r.failure_mode == min(
            r.mechanism_mttf_hours,
            key=lambda name: r.mechanism_mttf_hours[name],
        )

    def test_high_temp(self):
        from sc_neurocore.compiler.intelligence import predict_reliability

        r = predict_reliability(temperature_c=125.0)
        assert r.failure_mode == min(
            r.mechanism_mttf_hours,
            key=lambda name: r.mechanism_mttf_hours[name],
        )
        assert r.temp_accel > 1.0


class TestDiscoveryHook:
    def test_register_and_discover(self):
        from sc_neurocore.compiler.intelligence import (
            register_platform_hook,
            discover_platforms,
            _DISCOVERY_HOOKS,
        )
        from sc_neurocore.compiler.platforms import HardwareProfile

        def my_hook():
            return [
                HardwareProfile(
                    name="test_discovered_chip",
                    vendor="HookVendor",
                    family="HookFam",
                    platform_class="custom",
                    data_width=16,
                    fraction=8,
                    overflow="saturate",
                    rounding="nearest",
                )
            ]

        register_platform_hook(my_hook)
        found = discover_platforms()
        assert "test_discovered_chip" in found
        # Cleanup
        _DISCOVERY_HOOKS.pop()


class TestSIMDPack:
    """Test SIMD-accelerated pack_bitstream_numpy correctness."""

    def test_pack_numpy_matches_list_pack(self):
        """SIMD pack must produce identical output to list pack."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10_000).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    @pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 256, 1024, 4096])
    def test_pack_numpy_various_lengths(self, length):
        """SIMD pack handles all lengths including non-aligned."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, length).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    def test_pack_numpy_deterministic(self):
        """Same input -> same output."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 128, dtype=np.uint8)
        a = np.asarray(v3.pack_bitstream_numpy(bits))
        b = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(a, b)

    def test_pack_unpack_roundtrip(self):
        """Pack->unpack roundtrip preserves bits."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 2048).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        unpacked = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, np.asarray(unpacked))


class TestBranchlessLIF:
    """Test that branchless LIF step produces identical results."""

    def test_100_steps_constant_input(self):
        """Standard equivalence: same as equivalence suite."""
        lif = v3.FixedPointLif()
        results = []
        for _ in range(100):
            s, v = lif.step(20, 256, 128, 0)
            results.append((s, v))
        assert len(results) == 100
        for s, v in results:
            assert s in (0, 1)
            assert isinstance(v, (int, np.integer))

    def test_batch_matches_step_by_step(self):
        """batch_lif_run must match step-by-step execution."""
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_refractory_period(self):
        """Refractory behavior preserved under branchless mask."""
        spikes, _ = v3.batch_lif_run(200, 20, 256, 200, refractory_period=5)
        spikes_arr = np.asarray(spikes)
        spike_indices = np.where(spikes_arr == 1)[0]
        for idx in spike_indices:
            for ref_step in range(1, 6):
                if idx + ref_step < len(spikes_arr):
                    assert spikes_arr[idx + ref_step] == 0, (
                        f"Spike during refractory at step {idx + ref_step}"
                    )


class TestMultiNeuronBatch:
    """Test parallel multi-neuron LIF batch."""

    def test_shape_and_dtype(self):
        """Output shape is (n_neurons, n_steps)."""
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        assert np.asarray(spikes).shape == (10, 100)
        assert np.asarray(voltages).shape == (10, 100)

    def test_matches_sequential(self):
        """Parallel multi-neuron must match N sequential single-neuron runs."""
        n_neurons = 8
        n_steps = 500
        i_values = [64, 96, 128, 160, 192, 224, 100, 140]
        currents = np.array(i_values, dtype=np.int16)

        sequential_spikes = []
        for i_t in i_values:
            s, _ = v3.batch_lif_run(n_steps, 20, 256, i_t)
            sequential_spikes.append(np.asarray(s))

        par_spikes, _ = v3.batch_lif_run_multi(n_neurons, n_steps, 20, 256, currents)
        par_arr = np.asarray(par_spikes)

        for ni in range(n_neurons):
            np.testing.assert_array_equal(
                par_arr[ni], sequential_spikes[ni], err_msg=f"Neuron {ni} mismatch"
            )

    def test_deterministic(self):
        """Same inputs -> same outputs."""
        currents = np.full(4, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        s2, v2 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))


class TestRayonThreshold:
    """Test that rayon threshold does not change forward_fast outputs."""

    def test_forward_fast_determinism(self):
        """forward_fast with small inputs (below threshold) stays deterministic."""
        layer = v3.DenseLayer(16, 8, 1024)
        inputs = [0.5] * 16
        a = layer.forward_fast(inputs, seed=42)
        b = layer.forward_fast(inputs, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_forward_fast_consistent_across_sizes(self):
        """forward_fast produces valid outputs for various input sizes."""
        for n_in in [4, 16, 64, 128, 256]:
            layer = v3.DenseLayer(n_in, 8, 1024)
            inputs = [0.5] * n_in
            result = layer.forward_fast(inputs, seed=42)
            assert len(result) == 8
            for val in result:
                assert 0.0 <= val <= float(n_in), f"Out of range: {val}"


class TestPhase10Version:
    def test_version(self):
        _assert_engine_version_matches_core()


class TestSIMDFusedAndPopcount:
    """Verify SIMD fused AND+popcount preserves dense behavior."""

    def test_dense_forward_unchanged(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out1 = layer.forward(inputs, seed=123)
        out2 = layer.forward(inputs, seed=123)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= x <= 8.0 for x in out1)

    def test_dense_prepacked_unchanged(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_determinism(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=77)
        out2 = layer.forward_fast(inputs, seed=77)
        np.testing.assert_array_equal(out1, out2)


class TestSIMDBernoulliEncode:
    """Verify SIMD Bernoulli encoder statistical correctness and determinism."""

    def test_batch_encode_statistics(self):
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_batch_encode_determinism(self):
        probs = np.array([0.15, 0.35, 0.55, 0.75], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        b = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        np.testing.assert_array_equal(a, b)

    def test_dense_fast_correctness(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=22))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=22))
        assert high > low


class TestFlatWeightStorage:
    """Verify flat packed weight storage keeps API behavior unchanged."""

    def test_weight_roundtrip(self):
        layer = v3.DenseLayer(4, 3, 256, seed=42)
        weights = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float64,
        )
        layer.set_weights(weights.tolist())
        got = np.array(layer.get_weights(), dtype=np.float64)
        np.testing.assert_allclose(got, weights)

    def test_forward_equivalence_vs_prepacked(self):
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        seed = 31415
        packed = v3.batch_encode_numpy(probs, length=512, seed=seed)
        out_fast = np.asarray(layer.forward_fast(probs.tolist(), seed=seed), dtype=np.float64)
        out_prepacked = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)
        np.testing.assert_allclose(out_fast, out_prepacked)


class TestZeroAllocLIF:
    """Verify pre-allocated LIF batch outputs stay correct."""

    def test_batch_lif_unchanged(self):
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_batch_lif_multi_unchanged(self):
        n_steps = 500
        currents = np.array([64, 96, 128, 160, 192, 224, 100, 140], dtype=np.int16)
        sequential_spikes = []
        for i_t in currents:
            spikes, _ = v3.batch_lif_run(n_steps, 20, 256, int(i_t))
            sequential_spikes.append(np.asarray(spikes))

        spikes_multi, _ = v3.batch_lif_run_multi(len(currents), n_steps, 20, 256, currents)
        spikes_multi = np.asarray(spikes_multi)
        for idx in range(len(currents)):
            np.testing.assert_array_equal(spikes_multi[idx], sequential_spikes[idx])

    def test_batch_lif_multi_shape(self):
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        spikes_arr = np.asarray(spikes)
        voltages_arr = np.asarray(voltages)
        assert spikes_arr.shape == (10, 100)
        assert voltages_arr.shape == (10, 100)
        assert spikes_arr.dtype == np.int32
        assert voltages_arr.dtype == np.int16

    def test_batch_lif_varying_unchanged(self):
        currents = np.array([120, 128, 136, 150, 160, 100, 80, 140], dtype=np.int16)
        noises = np.array([0, 1, -1, 2, -2, 0, 1, -1], dtype=np.int16)

        lif = v3.FixedPointLif()
        ref_spikes, ref_voltages = [], []
        for i_t, n_t in zip(currents, noises):
            s, v = lif.step(20, 256, int(i_t), int(n_t))
            ref_spikes.append(s)
            ref_voltages.append(v)

        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20,
            gain_k=256,
            currents=currents,
            noises=noises,
        )
        np.testing.assert_array_equal(np.asarray(spikes), np.array(ref_spikes, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(voltages), np.array(ref_voltages, dtype=np.int16))


class TestPhase11Version:
    def test_version(self):
        _assert_engine_version_matches_core()


class TestFusedKernel:
    """Verify fused encode+AND+popcount behavior and determinism."""

    def test_fused_matches_forward_fast(self):
        """Fused forward_fast output matches prepacked materialized encode path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        seed = 123

        fused = np.asarray(layer.forward_fast(inputs.tolist(), seed=seed), dtype=np.float64)
        packed = v3.batch_encode_numpy(inputs, length=512, seed=seed)
        materialized = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)

        np.testing.assert_array_equal(fused, materialized)

    def test_fused_determinism(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=777)
        out2 = layer.forward_fast(inputs, seed=777)
        np.testing.assert_array_equal(out1, out2)

    def test_fused_statistical_correctness(self):
        layer = v3.DenseLayer(16, 8, 2048, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=42))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=42))
        assert high > low


class TestFastPRNG:
    """Verify xoshiro-backed fast paths remain deterministic and statistically sane."""

    def test_xoshiro_determinism(self):
        probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        b = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        np.testing.assert_array_equal(a, b)

    def test_xoshiro_statistical_quality(self):
        probs = np.array([0.35], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=1337)
        count = sum(int(w).bit_count() for w in packed[0])
        measured = count / 10_000
        assert abs(measured - 0.35) < 0.03

    def test_forward_fast_determinism_new(self):
        layer = v3.DenseLayer(12, 6, 1024, seed=42)
        inputs = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        a = layer.forward_fast(inputs.tolist(), seed=98765)
        b = layer.forward_fast(inputs.tolist(), seed=98765)
        np.testing.assert_array_equal(a, b)


class TestBatchForward:
    """Verify batched forward API correctness, shape and determinism."""

    def test_batch_vs_sequential(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.random.RandomState(42).uniform(0, 1, (10, 8)).astype(np.float64)
        seed = 555

        batched = np.asarray(layer.forward_batch_numpy(inputs, seed=seed), dtype=np.float64)

        sequential_rows = []
        for sample_idx, row in enumerate(inputs):
            sample_seed = seed + sample_idx * 1_000_000
            sequential_rows.append(layer.forward_fast(row.tolist(), seed=sample_seed))
        sequential = np.asarray(sequential_rows, dtype=np.float64)

        np.testing.assert_array_equal(batched, sequential)

    def test_batch_shape(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(1).uniform(0, 1, (25, 16)).astype(np.float64)
        out = np.asarray(layer.forward_batch_numpy(inputs, seed=100))
        assert out.shape == (25, 8)

    def test_batch_determinism(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.random.RandomState(7).uniform(0, 1, (12, 16)).astype(np.float64)
        a = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        b = np.asarray(layer.forward_batch_numpy(inputs, seed=101))
        np.testing.assert_array_equal(a, b)

    def test_batch_numpy_output(self):
        layer = v3.DenseLayer(4, 2, 256, seed=42)
        inputs = np.random.RandomState(9).uniform(0, 1, (3, 4)).astype(np.float64)
        out = layer.forward_batch_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64


class TestPhase12Version:
    def test_version(self):
        _assert_engine_version_matches_core()


class TestForwardNumpy:
    """Tests for single-call numpy dense forward."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8,)
        assert out.dtype == np.float64

    def test_output_range(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.3] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert np.all(out >= 0.0)
        assert np.all(out <= 16.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_matches_forward_fast(self):
        """forward_numpy should match forward_fast with same seed."""
        layer = v3.DenseLayer(8, 4, 256, seed=42)
        inputs_list = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        inputs_np = np.array(inputs_list, dtype=np.float64)
        out_fast = layer.forward_fast(inputs_list, seed=42)
        out_numpy = layer.forward_numpy(inputs_np, seed=42)
        np.testing.assert_allclose(out_numpy, out_fast)

    def test_wrong_input_length(self):
        layer = v3.DenseLayer(8, 4, 256)
        inputs = np.array([0.5] * 7, dtype=np.float64)
        with pytest.raises(ValueError):
            layer.forward_numpy(inputs)

    def test_different_seed_different_output(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.array([0.5] * 8, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=200)
        assert not np.array_equal(out1, out2)


class TestParallelBatchEncodeNumpy:
    """Tests for parallel batch_encode_numpy."""

    def test_shape_and_dtype(self):
        probs = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        assert packed.shape == (3, 16)
        assert packed.dtype == np.uint64

    def test_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed(self):
        probs = np.array([0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=1024, seed=1)
        p2 = v3.batch_encode_numpy(probs, length=1024, seed=2)
        assert not np.array_equal(p1, p2)

    def test_popcount_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_pipeline_encode_then_forward(self):
        """batch_encode_numpy -> forward_prepacked remains valid."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 4.0 for v in out)

    def test_empty_probs(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape[0] == 0


class TestPhase8Version:
    def test_version_is_current(self):
        _assert_engine_version_matches_core()


class TestFastBernoulli:
    """Tests for byte-threshold Bernoulli in forward_fast and batch_encode_numpy."""

    def test_forward_fast_deterministic(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=100)
        out2 = layer.forward_fast(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_forward_fast_output_range(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.3] * 8
        out = layer.forward_fast(inputs, seed=42)
        assert all(v >= 0.0 for v in out)

    def test_forward_fast_statistical_sanity(self):
        """forward_fast output should correlate with input probability."""
        layer = v3.DenseLayer(8, 4, 2048, seed=42)
        low_out = np.mean(layer.forward_fast([0.1] * 8, seed=42))
        high_out = np.mean(layer.forward_fast([0.9] * 8, seed=42))
        assert high_out > low_out, "Higher input probs should give higher output"

    def test_batch_encode_numpy_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_batch_encode_numpy_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.04
        assert abs(pc1 / 10_000 - 0.75) < 0.04


class TestFusedAndPopcount:
    """Tests verifying fused AND+popcount produces same results as before."""

    def test_forward_matches_reference(self):
        """forward() output should still be valid (range + deterministic)."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1]
        out1 = layer.forward(inputs, seed=42)
        out2 = layer.forward(inputs, seed=42)
        np.testing.assert_array_equal(out1, out2)
        assert all(v >= 0.0 for v in out1)

    def test_prepacked_deterministic(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        np.testing.assert_array_equal(out1, out2)


class TestZeroCopyPrepackedNumpy:
    """Tests for forward_prepacked_numpy (true zero-copy path)."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked_numpy(packed)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64

    def test_matches_forward_prepacked(self):
        """Zero-copy numpy path must match the existing prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_wrong_n_inputs(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((3, 16), dtype=np.uint64)  # 3 inputs, need 4
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_wrong_word_count(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((4, 10), dtype=np.uint64)  # 10 words, need 16
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_pipeline_encode_then_zero_copy(self):
        """Full pipeline: batch_encode_numpy -> forward_prepacked_numpy."""
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        probs = np.random.uniform(0, 1, 16)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        out = layer.forward_prepacked_numpy(packed)
        assert out.shape == (8,)
        assert np.all(out >= 0.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(4, 2, 512, seed=42)
        probs = np.array([0.5] * 4, dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=42)
        out1 = layer.forward_prepacked_numpy(packed)
        out2 = layer.forward_prepacked_numpy(packed)
        np.testing.assert_array_equal(out1, out2)


class TestSetNumThreads:
    """Tests for rayon thread pool configuration."""

    def test_set_num_threads_does_not_crash(self):
        """Calling set_num_threads should not raise."""
        # Can only be set before global pool initialization. If initialized,
        # rayon returns an error, which is acceptable behavior.
        try:
            v3.set_num_threads(0)  # 0 = default
        except ValueError:
            pass


class TestPhase9Version:
    def test_version_is_current(self):
        _assert_engine_version_matches_core()
