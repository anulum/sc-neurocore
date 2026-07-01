# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the equivalence miter builder

"""Unit tests for sequential-equivalence miter construction (no external tools)."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.equivalence_miter import (
    MiterPort,
    _eval_width_expr,
    build_equivalence_miter,
    parse_module_interface,
)

_LIF_REFERENCE = """
module sc_lif_reference #(
    parameter integer DATA_WIDTH = 16,
    parameter integer FRACTION = 8,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = (1 << FRACTION)
)(
    input wire                            clk,
    input wire                            rst_n,
    input wire signed [DATA_WIDTH-1:0]    leak_k,
    input wire signed [DATA_WIDTH-1:0]    I_t,
    output reg                            spike_out,
    output reg signed [DATA_WIDTH-1:0]    v_out
);
endmodule
"""


def _lif_ports() -> list[MiterPort]:
    return [
        MiterPort("clk", 1, False, "input"),
        MiterPort("rst_n", 1, False, "input"),
        MiterPort("leak_k", 16, True, "input"),
        MiterPort("spike_out", 1, False, "output"),
        MiterPort("v_out", 16, True, "output"),
    ]


class TestParseModuleInterface:
    """Parsing the ANSI port interface, resolving parameter-dependent widths."""

    def test_parses_directions_widths_and_signedness(self) -> None:
        ports = parse_module_interface(
            _LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 16}
        )
        by_name = {p.name: p for p in ports}
        assert by_name["clk"] == MiterPort("clk", 1, False, "input")
        assert by_name["rst_n"] == MiterPort("rst_n", 1, False, "input")
        assert by_name["leak_k"] == MiterPort("leak_k", 16, True, "input")
        assert by_name["I_t"] == MiterPort("I_t", 16, True, "input")
        assert by_name["spike_out"] == MiterPort("spike_out", 1, False, "output")
        assert by_name["v_out"] == MiterPort("v_out", 16, True, "output")

    def test_declaration_order_is_preserved(self) -> None:
        ports = parse_module_interface(
            _LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 16}
        )
        assert [p.name for p in ports] == [
            "clk",
            "rst_n",
            "leak_k",
            "I_t",
            "spike_out",
            "v_out",
        ]

    def test_width_expression_uses_parameter_value(self) -> None:
        ports = parse_module_interface(_LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 8})
        assert {p.name: p.width for p in ports}["v_out"] == 8

    def test_missing_module_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            parse_module_interface(_LIF_REFERENCE, "no_such_module")

    def test_unknown_width_parameter_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            parse_module_interface(_LIF_REFERENCE, "sc_lif_reference")

    def test_module_without_ports_raises(self) -> None:
        with pytest.raises(ValueError, match="no ports|port list"):
            parse_module_interface("module empty(); endmodule", "empty")

    def test_shift_and_arithmetic_in_width(self) -> None:
        src = "module m(input wire [ (2*4) - 1 : 0 ] a, output wire [1<<3:0] b); endmodule"
        ports = parse_module_interface(src, "m")
        widths = {p.name: p.width for p in ports}
        assert widths["a"] == 8  # (2*4) - 1 downto 0 -> 8 bits
        assert widths["b"] == 9  # (1<<3)=8 downto 0 -> 9 bits


class TestBuildEquivalenceMiter:
    """Rendering the miter Verilog."""

    def test_instantiates_both_modules_with_shared_inputs(self) -> None:
        miter = build_equivalence_miter("sc_lif_neuron", "sc_lif_reference", _lif_ports())
        assert "sc_lif_neuron" in miter
        assert "sc_lif_reference" in miter
        assert "dut (" in miter
        assert "ref_model (" in miter
        # Free (non-reset) inputs are exposed as top-level miter inputs.
        assert "input wire signed [15:0] leak_k" in miter
        # The clock is a top-level input; reset is derived, not exposed.
        assert "input wire clk" in miter
        assert "input wire rst_n" not in miter

    def test_reset_counter_holds_reset_then_releases(self) -> None:
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports(), reset_cycles=3)
        assert "localparam integer RESET_CYCLES = 3;" in miter
        assert "reg [7:0] rst_cnt = 0;" in miter
        assert "wire rst_n = (rst_cnt >= RESET_CYCLES);" in miter
        # No simulation-only constructs that would over-constrain the checker.
        assert "initial" not in miter
        assert "#5" not in miter

    def test_asserts_every_output_pair_after_reset(self) -> None:
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports())
        assert "assert(spike_out_dut == spike_out_ref);" in miter
        assert "assert(v_out_dut == v_out_ref);" in miter
        assert "if (rst_n) begin" in miter

    def test_parameter_overrides_applied_per_instance(self) -> None:
        miter = build_equivalence_miter(
            "dut_mod",
            "ref_mod",
            _lif_ports(),
            dut_params={"DATA_WIDTH": 16, "REFRACTORY_PERIOD": 0},
            ref_params={"DATA_WIDTH": 16},
        )
        assert ".REFRACTORY_PERIOD(0)" in miter
        assert ".DATA_WIDTH(16)" in miter

    def test_output_comparison_wires_declared(self) -> None:
        miter = build_equivalence_miter("dut_mod", "ref_mod", _lif_ports())
        assert "wire signed [15:0] v_out_dut" in miter
        assert "wire signed [15:0] v_out_ref" in miter
        assert "wire spike_out_dut" in miter

    def test_same_top_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            build_equivalence_miter("same", "same", _lif_ports())

    def test_missing_clock_rejected(self) -> None:
        ports = [p for p in _lif_ports() if p.name != "clk"]
        with pytest.raises(ValueError, match="clock port"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_missing_reset_rejected(self) -> None:
        ports = [p for p in _lif_ports() if p.name != "rst_n"]
        with pytest.raises(ValueError, match="reset port"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_no_outputs_rejected(self) -> None:
        ports = [p for p in _lif_ports() if p.direction != "output"]
        with pytest.raises(ValueError, match="at least one output"):
            build_equivalence_miter("dut_mod", "ref_mod", ports)

    def test_zero_reset_cycles_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            build_equivalence_miter("dut_mod", "ref_mod", _lif_ports(), reset_cycles=0)

    def test_custom_clock_and_reset_names(self) -> None:
        ports = [
            MiterPort("clock_i", 1, False, "input"),
            MiterPort("resetn_i", 1, False, "input"),
            MiterPort("q", 8, False, "output"),
        ]
        miter = build_equivalence_miter("d", "r", ports, clock="clock_i", reset_n="resetn_i")
        assert "input wire clock_i" in miter
        assert "wire resetn_i = (rst_cnt >= RESET_CYCLES);" in miter
        assert "always @(posedge clock_i)" in miter


class TestWidthEvaluator:
    """The restricted arithmetic evaluator behind parameter-dependent widths."""

    @pytest.mark.parametrize(
        ("expr", "params", "expected"),
        [
            ("7", {}, 7),
            ("W - 1", {"W": 32}, 31),
            ("2 + 3", {}, 5),
            ("2 * 4", {}, 8),
            ("8 // 2", {}, 4),
            ("1 << 4", {}, 16),
            ("16 >> 2", {}, 4),
            ("- -8", {}, 8),
            ("W // 2 + 1", {"W": 16}, 9),
        ],
    )
    def test_operators(self, expr: str, params: dict[str, int], expected: int) -> None:
        assert _eval_width_expr(expr, params) == expected

    def test_non_integer_literal_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-integer literal"):
            _eval_width_expr("1.5", {})

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            _eval_width_expr("MISSING", {})

    def test_unsupported_operator_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported width expression"):
            _eval_width_expr("2 ** 3", {})


class TestHeaderErrors:
    """Defensive parsing paths in the module-header locator."""

    def test_malformed_parameter_block(self) -> None:
        with pytest.raises(ValueError, match="malformed parameter block"):
            parse_module_interface("module m #", "m")

    def test_no_port_list_at_all(self) -> None:
        with pytest.raises(ValueError, match="port list .* not found"):
            parse_module_interface("module m; endmodule", "m")

    def test_unterminated_port_list(self) -> None:
        with pytest.raises(ValueError, match="unterminated port list"):
            parse_module_interface("module m ( input wire a", "m")

    def test_non_positive_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-positive width"):
            parse_module_interface("module m(input wire [0:5] a); endmodule", "m")

    def test_parameter_block_before_ports_is_skipped(self) -> None:
        src = "module m #(parameter W = 4)(input wire [W-1:0] a, output wire b); endmodule"
        ports = parse_module_interface(src, "m", params={"W": 4})
        assert {p.name: p.width for p in ports} == {"a": 4, "b": 1}
