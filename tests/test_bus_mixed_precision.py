# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Tests for bus interface generation + mixed-precision

"""Tests for AXI4-Lite / Wishbone wrappers and mixed-precision solver."""

from __future__ import annotations

import pytest

from sc_neurocore.hdl_gen.bus_interface import (
    generate_bus_wrapper,
    generate_register_map,
)
from sc_neurocore.compiler.mixed_precision import (
    MixedPrecisionSpec,
    PrecisionConfig,
    BlockFloatingPrecisionConfig,
    from_preset,
    solve_precision,
    PRECISION_PRESETS,
)


# ═══════════════════════════════════════════════════════════════════════
# Bus Interface Tests
# ═══════════════════════════════════════════════════════════════════════

LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}


class TestAXI4Lite:
    """Test AXI4-Lite wrapper generation."""

    def test_generates_module(self) -> None:
        """Should produce a valid Verilog module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "module sc_lif_axi_lite" in v
        assert "endmodule" in v

    def test_has_axi_ports(self) -> None:
        """Should include all AXI4-Lite signal names."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        for sig in [
            "S_AXI_ACLK",
            "S_AXI_ARESETN",
            "S_AXI_AWADDR",
            "S_AXI_WDATA",
            "S_AXI_RDATA",
            "S_AXI_BRESP",
        ]:
            assert sig in v, f"Missing AXI signal: {sig}"

    def test_has_interrupt(self) -> None:
        """Should export spike interrupt."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "irq_spike" in v

    def test_has_neuron_instance(self) -> None:
        """Should instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers(self) -> None:
        """Each parameter should have a register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_p_v_rest" in v
        assert "reg_p_v_thresh" in v
        assert "reg_p_tau_m" in v

    def test_has_spike_counter(self) -> None:
        """Should include a spike counter register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_spike_count" in v

    def test_control_register(self) -> None:
        """Should have enable and reset bits in control register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_ctrl" in v
        assert "reg_ctrl[0]" in v  # enable
        assert "reg_ctrl[1]" in v  # reset


class TestWishbone:
    """Test Wishbone B4 wrapper generation."""

    def test_generates_module(self) -> None:
        """Should produce a valid Verilog module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "module sc_lif_wb" in v
        assert "endmodule" in v

    def test_has_wishbone_ports(self) -> None:
        """Should include all Wishbone signal names."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        for sig in [
            "wb_clk_i",
            "wb_rst_i",
            "wb_adr_i",
            "wb_dat_i",
            "wb_dat_o",
            "wb_we_i",
            "wb_stb_i",
            "wb_cyc_i",
            "wb_ack_o",
        ]:
            assert sig in v, f"Missing Wishbone signal: {sig}"

    def test_has_interrupt(self) -> None:
        """Should export spike interrupt."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "irq_spike" in v

    def test_has_neuron_instance(self) -> None:
        """Should instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers(self) -> None:
        """Each parameter should have a register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "reg_p_v_rest" in v


class TestRegisterMap:
    """Test register map generation."""

    def test_standard_layout(self) -> None:
        """Standard registers should be at expected offsets."""
        rmap = generate_register_map(LIF_PARAMS)
        assert rmap["CTRL"] == 0
        assert rmap["I_T"] == 4
        assert rmap["SPIKE_COUNT"] == 8
        assert rmap["P_V_REST"] == 12

    def test_custom_base_address(self) -> None:
        """Base address should shift all registers."""
        rmap = generate_register_map(LIF_PARAMS, base_address=0x1000)
        assert rmap["CTRL"] == 0x1000
        assert rmap["I_T"] == 0x1004

    def test_invalid_bus(self) -> None:
        """Should raise on invalid bus protocol."""
        with pytest.raises(ValueError, match="Unsupported bus"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="spi")  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# Mixed-Precision Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPrecisionConfig:
    """Test the PrecisionConfig dataclass."""

    def test_q88_properties(self) -> None:
        """Q8.8 should expose sign-inclusive label and magnitude integer bits."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.int_bits == 7
        assert cfg.q_label == "Q8.8"
        assert cfg.resolution == pytest.approx(1 / 256)

    def test_unsigned(self) -> None:
        """Unsigned config should have min=0 and doubled positive range."""
        cfg = PrecisionConfig(16, 8, signed=False)
        assert cfg.min_value == 0.0
        assert cfg.max_value > 200

    def test_can_represent(self) -> None:
        """Range checking should work."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.can_represent(50.0)
        assert not cfg.can_represent(200.0)

    def test_encode(self) -> None:
        """Encoding should produce correct Q-format integers."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.encode(1.0) == 256
        assert cfg.encode(-1.0) == -256
        assert cfg.encode(0.5) == 128


class TestMixedPrecisionSpec:
    """Test the mixed-precision specification."""

    def test_total_bits(self) -> None:
        """Total bits should sum correctly."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        assert spec.total_bits == 24

    def test_variables(self) -> None:
        """Should list all variables."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        assert set(spec.variables) == {"v", "u"}

    def test_get(self) -> None:
        """Should retrieve config by name."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
            }
        )
        assert spec.get("v").data_width == 15
        assert spec.get("v").q_label == "Q7.8"

    def test_get_missing(self) -> None:
        """Should raise on missing variable."""
        spec = MixedPrecisionSpec({"v": PrecisionConfig(16, 8)})
        with pytest.raises(KeyError, match="not in"):
            spec.get("w")

    def test_summary(self) -> None:
        """Summary should be human-readable."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        s = spec.summary()
        assert "24 bits total" in s
        assert "Q8.8" in s
        assert "Q4.4" in s


class TestConstraintSolver:
    """Test the automatic precision constraint solver."""

    def test_basic_solve(self) -> None:
        """Should produce valid configs from bounds."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
        )
        assert spec.get("v").can_represent(-128)
        assert spec.get("v").can_represent(127)
        assert spec.get("u").can_represent(-10)
        assert spec.get("u").can_represent(10)

    def test_resolution_honoured(self) -> None:
        """Requested resolution should be achievable."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.001},
        )
        assert spec.get("v").resolution <= 0.001

    def test_budget_constraint(self) -> None:
        """Should reduce precision to fit bit budget."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
            max_total_bits=24,
        )
        assert spec.total_bits <= 24

    def test_alignment(self) -> None:
        """Byte alignment should round up data widths."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.01},
            align_to=8,
        )
        assert spec.get("v").data_width % 8 == 0

    def test_single_variable(self) -> None:
        """Should work with a single variable."""
        spec = solve_precision(
            bounds={"x": (0, 255)},
            min_resolution={"x": 0.1},
        )
        assert spec.get("x").can_represent(255)

    def test_mixed_ranges(self) -> None:
        """Variables with very different ranges get different widths."""
        spec = solve_precision(
            bounds={"v": (-32768, 32767), "flag": (0, 1)},
            min_resolution={"v": 0.01, "flag": 0.5},
        )
        assert spec.get("v").data_width > spec.get("flag").data_width


class TestFromPreset:
    """Test preset-based mixed-precision creation."""

    def test_basic(self) -> None:
        """Should create spec from named presets."""
        spec = from_preset({"v": "q88", "u": "q44"})
        assert spec.get("v").data_width == 16
        assert spec.get("u").data_width == 8

    def test_all_presets_valid(self) -> None:
        """Every preset in the registry should be valid."""
        for name, cfg in PRECISION_PRESETS.items():
            assert cfg.data_width > 0
            assert cfg.fraction >= 0
            assert cfg.fraction < cfg.data_width

    def test_unknown_preset(self) -> None:
        """Should raise on unknown preset."""
        with pytest.raises(KeyError, match="Unknown preset"):
            from_preset({"v": "q999"})

    def test_case_insensitive(self) -> None:
        """Preset lookup should be case-insensitive."""
        spec = from_preset({"v": "Q7.8"})
        assert spec.get("v").data_width == 15
        assert spec.get("v").q_label == "Q7.8"

    def test_block_floating_preset(self) -> None:
        """Block-floating presets should be materializable and typed."""
        spec = from_preset({"k": "bfp16e3x32"})
        cfg = spec.get("k")
        assert isinstance(cfg, BlockFloatingPrecisionConfig)
        assert cfg.mantissa_bits == 16
        assert cfg.exponent_bits == 3
        assert cfg.block_size == 32
        assert cfg.kind == "block_floating"
