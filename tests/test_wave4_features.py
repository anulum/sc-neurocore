# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851

"""Tests for Wave 4: BRAM auto-selection, thermal analysis, weight ROM, BRAM array."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# A. BRAM / Register Auto-Selection
# ═══════════════════════════════════════════════════════════════════════

class TestStorageRecommendation:
    """Tests for BRAM/register storage strategy."""

    def test_small_uses_registers(self):
        """≤64 neurons → registers."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(32, 16)
        assert rec.strategy == "registers"
        assert rec.total_bits == 32 * 16

    def test_medium_uses_bram(self):
        """65–16K neurons → BRAM."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(1024, 16)
        assert rec.strategy == "bram"
        assert rec.total_bits == 1024 * 16

    def test_large_with_uram(self):
        """≥16K neurons with URAM → URAM."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(20000, 16, has_uram=True)
        assert rec.strategy == "uram"
        assert rec.uram_used >= 1

    def test_large_without_uram_uses_bram(self):
        """Large without URAM → falls back to BRAM."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(20000, 16, has_uram=False)
        assert rec.strategy == "bram"

    def test_custom_threshold(self):
        """Custom register threshold."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(100, 16, register_threshold=128)
        assert rec.strategy == "registers"

    def test_bram_18k_for_small(self):
        """Small BRAM uses 18Kb tile."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(128, 16)  # 2048 bits, fits in 18Kb
        assert rec.strategy == "bram"
        assert rec.bram_18k_used == 1
        assert rec.bram_36k_used == 0

    def test_bram_36k_for_large(self):
        """Larger BRAM uses 36Kb tiles."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(4096, 16)  # 65536 bits
        assert rec.strategy == "bram"
        assert rec.bram_36k_used >= 1

    def test_reason_populated(self):
        """Reason string is non-empty."""
        from sc_neurocore.compiler.advanced_features import storage_recommendation
        rec = storage_recommendation(10, 16)
        assert len(rec.reason) > 0


class TestBRAMArray:
    """Tests for BRAM-backed neuron array generation."""

    def test_basic_array(self):
        """Default array generates valid Verilog."""
        from sc_neurocore.compiler.advanced_features import generate_bram_array
        v = generate_bram_array()
        assert "module sc_neuron_array" in v
        assert "state_bram" in v
        assert "ram_style" in v
        assert "endmodule" in v

    def test_custom_count(self):
        """Custom neuron count."""
        from sc_neurocore.compiler.advanced_features import generate_bram_array
        v = generate_bram_array(neuron_count=256)
        assert "[0:255]" in v

    def test_custom_module_name(self):
        """Custom module name."""
        from sc_neurocore.compiler.advanced_features import generate_bram_array
        v = generate_bram_array(module_name="my_array")
        assert "module my_array" in v

    def test_spike_output(self):
        """Array has spike output ports."""
        from sc_neurocore.compiler.advanced_features import generate_bram_array
        v = generate_bram_array()
        assert "spike_out" in v
        assert "spike_neuron_id" in v
        assert "tick_done" in v


# ═══════════════════════════════════════════════════════════════════════
# B. Thermal-Aware Compilation
# ═══════════════════════════════════════════════════════════════════════

class TestThermalAnalysis:
    """Tests for thermal estimation and derating."""

    def test_basic_thermal(self):
        """Basic thermal analysis returns valid fields."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(100.0, 500.0)
        assert t.junction_temp_c > 25.0
        assert t.derated_freq_mhz > 0
        assert t.thermal_safe
        assert t.hotspot_risk in ("none", "low", "medium", "high")

    def test_low_power_safe(self):
        """Low power design is thermally safe."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(0.1, 100.0)
        assert t.thermal_safe
        assert t.delta_t_c < 1.0

    def test_high_power_derating(self):
        """High power causes frequency derating."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(10000.0, 500.0)  # 10W
        assert t.junction_temp_c > 85.0
        assert t.derated_freq_mhz < 500.0

    def test_extreme_power_unsafe(self):
        """Extreme power exceeds junction limit."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(50000.0, 500.0)  # 50W
        assert not t.thermal_safe

    def test_dsp_hotspot(self):
        """Many DSPs in one column → high hotspot risk."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=1)
        assert t.hotspot_risk == "high"

    def test_dsp_spread(self):
        """DSPs spread across columns → lower risk."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=10)
        assert t.hotspot_risk in ("none", "low")

    def test_small_process_more_derating(self):
        """7nm process derates more than 28nm."""
        from sc_neurocore.compiler.advanced_features import thermal_analysis
        t7 = thermal_analysis(5000.0, 500.0, process_nm=7)
        t28 = thermal_analysis(5000.0, 500.0, process_nm=28)
        assert t7.derated_freq_mhz < t28.derated_freq_mhz


class TestThermalConstraints:
    """Tests for thermal constraint generation."""

    def test_basic_constraints(self):
        """Thermal constraints include derated clock."""
        from sc_neurocore.compiler.advanced_features import (
            thermal_analysis, generate_thermal_constraints,
        )
        t = thermal_analysis(100.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "create_clock" in xdc
        assert "Derated frequency" in xdc

    def test_hotspot_constraints(self):
        """High hotspot risk adds DSP spreading."""
        from sc_neurocore.compiler.advanced_features import (
            thermal_analysis, generate_thermal_constraints,
        )
        t = thermal_analysis(100.0, 500.0, mul_count=25, dsp_columns=1)
        xdc = generate_thermal_constraints("sc_hh", t)
        assert "DSP spreading" in xdc

    def test_unsafe_warning(self):
        """Unsafe temperature adds warning."""
        from sc_neurocore.compiler.advanced_features import (
            thermal_analysis, generate_thermal_constraints,
        )
        t = thermal_analysis(50000.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "WARNING" in xdc


# ═══════════════════════════════════════════════════════════════════════
# C. Weight ROM Generation
# ═══════════════════════════════════════════════════════════════════════

class TestWeightROM:
    """Tests for synaptic weight ROM generation."""

    def test_verilog_rom(self):
        """Verilog ROM module."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[1, 2], [3, 4]]
        v = generate_weight_rom(w)
        assert "module sc_weight_rom" in v
        assert "case" in v
        assert "endmodule" in v

    def test_coe_format(self):
        """Xilinx .coe format."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[10, 20], [30, 40]]
        coe = generate_weight_rom(w, output_format="coe")
        assert "memory_initialization_radix=16" in coe
        assert "memory_initialization_vector=" in coe

    def test_mif_format(self):
        """Intel .mif format."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[10, 20], [30, 40]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "WIDTH=16" in mif
        assert "DEPTH=4" in mif
        assert "CONTENT BEGIN" in mif
        assert "END;" in mif

    def test_custom_module_name(self):
        """Custom ROM module name."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[1, 2]]
        v = generate_weight_rom(w, module_name="my_weights")
        assert "module my_weights" in v

    def test_correct_entry_count(self):
        """Correct number of entries in ROM."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[1, 2, 3], [4, 5, 6]]
        mif = generate_weight_rom(w, output_format="mif")
        assert "DEPTH=6" in mif

    def test_data_width_propagates(self):
        """Custom data width propagates."""
        from sc_neurocore.compiler.advanced_features import generate_weight_rom
        w = [[1]]
        mif = generate_weight_rom(w, data_width=8, output_format="mif")
        assert "WIDTH=8" in mif
