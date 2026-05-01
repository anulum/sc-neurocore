# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851

"""Tests for Wave 2 features: SymbiYosys, RISC-V, DVS, SLR, MXFP, certification."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# A. New Hardware Profiles (Wave 1)
# ═══════════════════════════════════════════════════════════════════════

class TestWave1Profiles:
    """Verify all 12 new hardware profiles are registered."""

    @pytest.mark.parametrize("name", [
        "loihi3", "northpole", "innatera_pulsar",
        "versal_ai_edge", "proasic3", "trion", "titanium",
        "gowin_arora_v", "intel_agilex5",
        "nvidia_dla", "mediatek_apu", "aws_inferentia",
    ])
    def test_profile_exists(self, name):
        """Profile is registered and retrievable."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0
        assert p.fraction >= 0
        assert p.vendor

    def test_total_profiles_at_least_77(self):
        """Total registry should have at least 77 profiles."""
        from sc_neurocore.compiler.hardware_profiles import list_profiles
        assert len(list_profiles()) >= 77

    def test_loihi3_is_neuromorphic(self):
        """Loihi 3 should be in the neuromorphic class."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile("loihi3")
        assert p.platform_class == "neuromorphic"
        assert p.data_width == 32
        assert p.overflow == "wrap"

    def test_versal_ai_edge_dsp58(self):
        """Versal AI Edge should use DSP58 with 27x24 multiplier."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile("versal_ai_edge")
        assert p.dsp_block == "DSP58"
        assert p.dsp_mult_a == 27
        assert p.dsp_mult_b == 24
        assert p.max_freq_mhz == 900


# ═══════════════════════════════════════════════════════════════════════
# B. SymbiYosys Formal Proof Flow
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# C. RISC-V Driver + FreeRTOS / Zephyr
# ═══════════════════════════════════════════════════════════════════════

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
        assert "sc_lif_get_spikes" in c
        assert "sc_lif_encode" in c
        assert "sc_lif_set_tau" in c
        assert "sc_lif_set_vth" in c
        assert "volatile" in c

    def test_freertos_template(self):
        """FreeRTOS template includes task and timer."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver
        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="freertos")
        assert 'FreeRTOS.h' in c
        assert "xTaskCreate" in c
        assert "sc_lif_tick" in c
        assert "sc_lif_start_rtos" in c
        assert "vTaskDelay" in c

    def test_zephyr_template(self):
        """Zephyr template includes thread and K_THREAD_DEFINE."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver
        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="zephyr")
        assert "zephyr/kernel.h" in c
        assert "K_THREAD_DEFINE" in c
        assert "k_msleep" in c

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


# ═══════════════════════════════════════════════════════════════════════
# D. DVS Event-Camera → AER Bridge
# ═══════════════════════════════════════════════════════════════════════

class TestDVSBridge:
    """Tests for DVS→AER bridge Verilog generation."""

    def test_basic_bridge(self):
        """Default bridge generates valid Verilog."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
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
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        v = generate_dvs_aer_bridge(addr_width=20, timestamp_width=48)
        assert "[19:0]" in v
        assert "[47:0]" in v

    def test_custom_module_name(self):
        """Custom module name."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        v = generate_dvs_aer_bridge(module_name="my_dvs_bridge")
        assert "module my_dvs_bridge" in v

    def test_fifo_depth(self):
        """FIFO depth affects address widths."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        v = generate_dvs_aer_bridge(fifo_depth=128)
        assert "[0:127]" in v  # 128-deep FIFO

    def test_polarity_bit_included(self):
        """Polarity bit appears in ports."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        v = generate_dvs_aer_bridge(polarity_bit=True)
        assert "dvs_polarity" in v
        assert "aer_polarity" in v

    def test_overflow_flag(self):
        """FIFO overflow detection present."""
        from sc_neurocore.compiler.advanced_features import generate_dvs_aer_bridge
        v = generate_dvs_aer_bridge()
        assert "fifo_overflow" in v
        assert "overflow_r" in v


# ═══════════════════════════════════════════════════════════════════════
# E. Multi-Die SLR Placement
# ═══════════════════════════════════════════════════════════════════════

class TestSLRPlacement:
    """Tests for multi-die SLR constraint generation."""

    def test_single_slr(self):
        """Single SLR placement generates PBLOCK."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement, generate_slr_constraints,
        )
        xdc = generate_slr_constraints([
            SLRPlacement("neuron_array", slr=0),
        ])
        assert "create_pblock pblock_slr0" in xdc
        assert "SLR0" in xdc
        # No inter-SLR directives for single SLR
        assert "REGISTER_DUPLICATION" not in xdc

    def test_multi_slr_pipeline_regs(self):
        """Multi-SLR adds pipeline register directives."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement, generate_slr_constraints,
        )
        xdc = generate_slr_constraints([
            SLRPlacement("input_stage", slr=0),
            SLRPlacement("compute_stage", slr=1),
        ])
        assert "SLR0" in xdc
        assert "SLR1" in xdc
        assert "REGISTER_DUPLICATION" in xdc
        assert "set_max_delay" in xdc

    def test_no_pipeline_regs(self):
        """Opt-out of pipeline register insertion."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement, generate_slr_constraints,
        )
        xdc = generate_slr_constraints(
            [SLRPlacement("a", 0), SLRPlacement("b", 1)],
            insert_pipeline_regs=False,
        )
        assert "REGISTER_DUPLICATION" not in xdc

    def test_custom_pblock_name(self):
        """Custom PBLOCK name."""
        from sc_neurocore.compiler.deployment import (
            SLRPlacement, generate_slr_constraints,
        )
        xdc = generate_slr_constraints([
            SLRPlacement("core", slr=2, pblock_name="pblock_core"),
        ])
        assert "create_pblock pblock_core" in xdc

    def test_auto_pblock_name(self):
        """Auto-generated PBLOCK name from SLR index."""
        from sc_neurocore.compiler.deployment import SLRPlacement
        p = SLRPlacement("test", slr=3)
        assert p.pblock_name == "pblock_slr3"


# ═══════════════════════════════════════════════════════════════════════
# F. Block-FP / MXFP Encoding
# ═══════════════════════════════════════════════════════════════════════

class TestMXFP:
    """Tests for MXFP / Block-FP encoding/decoding."""

    def test_mxfp4_config(self):
        """MXFP4 config matches OCP spec."""
        from sc_neurocore.compiler.advanced_features import MXFP4
        assert MXFP4.element_bits == 4
        assert MXFP4.block_size == 32
        assert MXFP4.shared_exp_bits == 8
        assert MXFP4.label == "MXFP4"
        assert MXFP4.bits_per_block == 8 + 32 * 4  # 136

    def test_mxfp8_e4m3_config(self):
        """MXFP8 E4M3 config."""
        from sc_neurocore.compiler.advanced_features import MXFP8_E4M3
        assert MXFP8_E4M3.element_bits == 8
        assert MXFP8_E4M3.exp_bits == 4
        assert MXFP8_E4M3.mantissa_bits == 3

    def test_fp8_no_shared_exp(self):
        """IEEE FP8 has no shared exponent (block_size=1)."""
        from sc_neurocore.compiler.advanced_features import FP8_E4M3
        assert FP8_E4M3.block_size == 1
        assert FP8_E4M3.shared_exp_bits == 0

    def test_encode_decode_roundtrip_mxfp4(self):
        """MXFP4 encode→decode roundtrip preserves sign and order."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP4, mxfp_encode_block, mxfp_decode_block,
        )
        values = [float(i) / 32 for i in range(32)]
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        # Order preserved
        for i in range(1, len(decoded)):
            assert decoded[i] >= decoded[i - 1]

    def test_encode_all_zeros(self):
        """All-zero block returns zero exponent."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP4, mxfp_encode_block,
        )
        exp, encoded = mxfp_encode_block([0.0] * 32, MXFP4)
        assert exp == 0
        assert all(e == 0 for e in encoded)

    def test_block_size_mismatch_raises(self):
        """Wrong block size raises ValueError."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP4, mxfp_encode_block,
        )
        with pytest.raises(ValueError, match="Block size"):
            mxfp_encode_block([1.0, 2.0], MXFP4)

    def test_negative_values(self):
        """Negative values have sign bit set."""
        from sc_neurocore.compiler.advanced_features import (
            MXFP4, mxfp_encode_block, mxfp_decode_block,
        )
        values = [-1.0] * 32
        exp, encoded = mxfp_encode_block(values, MXFP4)
        decoded = mxfp_decode_block(exp, encoded, MXFP4)
        assert all(d < 0 for d in decoded)

    def test_mxfp6_exists(self):
        """MXFP6 config exists."""
        from sc_neurocore.compiler.advanced_features import MXFP6
        assert MXFP6.element_bits == 6


# ═══════════════════════════════════════════════════════════════════════
# G. Safety Certification Evidence
# ═══════════════════════════════════════════════════════════════════════

class TestCertificationEvidence:
    """Tests for safety-critical certification evidence generation."""

    def test_do254_xml(self):
        """DO-254 evidence generates valid XML structure."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem, generate_certification_evidence,
        )
        items = [
            CertificationItem("REQ-001", "No overflow", "sc_lif.v",
                              "sc_lif_sva.sv", "PASS"),
            CertificationItem("REQ-002", "Reset clears state", "sc_lif.v",
                              "test_reset", "PASS"),
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
            CertificationItem, generate_certification_evidence,
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
            CertificationItem, generate_certification_evidence,
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
            CertificationItem, generate_certification_evidence,
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
