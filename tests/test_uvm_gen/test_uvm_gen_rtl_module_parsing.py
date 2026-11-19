# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused RTL module-header and parameterised-width parsing tests

"""Verify RTLModule parsing across parameterised, AMBA AXI, and unpacked-array HDL."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from sc_neurocore.uvm_gen import (
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    ScoreboardConfig,
    UVMGenerator,
)
from sc_neurocore.uvm_gen._rtl import _eval_sv_int
from tools.benchmark_evidence_gate import tracked_files

REPO_ROOT = Path(__file__).resolve().parents[2]

_LIF_VERILOG = """\
module sc_lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter FRACTION = 8,
    parameter V_REST = 0,
    parameter V_THRESHOLD = 16'sd256,
    parameter REFRACTORY_PERIOD = 2
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire signed [15:0] leak_k,
    input  wire signed [15:0] gain_k,
    input  wire signed [15:0] I_t,
    input  wire signed [15:0] noise_in,
    output wire        spike_out,
    output wire signed [15:0] v_out
);
endmodule
"""

_DENSE_VERILOG = """\
module sc_dense_layer_core #(
    parameter NUM_NEURONS = 10
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [7:0] input_bus,
    output logic [7:0] output_bus
);
endmodule
"""


def _lif_module() -> RTLModule:
    return RTLModule.from_verilog_source(_LIF_VERILOG)


def _dense_module() -> RTLModule:
    return RTLModule.from_verilog_source(_DENSE_VERILOG)


class TestRTLModuleParsing:
    """Verify baseline and parameterised Verilog/SystemVerilog header parsing."""

    def test_parse_module_name(self) -> None:
        """Extract the exact RTL module identifier from a parameterised header."""
        rtl = _lif_module()
        assert rtl.name == "sc_lif_neuron"

    def test_parse_ports(self) -> None:
        """Parse all declared clock, reset, data-input, and output ports."""
        rtl = _lif_module()
        assert len(rtl.ports) == 8

    def test_parse_params_strip_trailing_commas(self) -> None:
        """Retain clean parameter default strings without trailing delimiter commas."""
        rtl = _lif_module()
        by_name = {p.name: p.value for p in rtl.params}
        assert by_name["DATA_WIDTH"] == "16"
        assert by_name["FRACTION"] == "8"
        assert by_name["V_THRESHOLD"] == "16'sd256"

    def test_clock_detection(self) -> None:
        """Identify scalar clock ports matching standard naming conventions."""
        rtl = _lif_module()
        assert rtl.clock_port is not None
        assert rtl.clock_port.name == "clk"

    def test_reset_detection(self) -> None:
        """Identify scalar reset ports matching standard naming conventions."""
        rtl = _lif_module()
        assert rtl.reset_port is not None
        assert rtl.reset_port.name == "rst_n"

    def test_input_ports_exclude_clock_reset(self) -> None:
        """Exclude clock and reset nets from generated stimulus input ports."""
        rtl = _lif_module()
        names = [p.name for p in rtl.input_ports]
        assert "clk" not in names
        assert "rst_n" not in names
        assert "I_t" in names

    def test_output_ports(self) -> None:
        """Collect all output ports for monitor, scoreboard, and coverage emission."""
        rtl = _lif_module()
        names = [p.name for p in rtl.output_ports]
        assert "spike_out" in names
        assert "v_out" in names

    def test_signed_ports(self) -> None:
        """Preserve the signedness qualifier on signed vector declarations."""
        rtl = _lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.is_signed is True

    def test_port_width(self) -> None:
        """Compute packed vector bit widths from MSB and LSB bounds."""
        rtl = _lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.width == 16

    def test_total_input_bits(self) -> None:
        """Sum data-input bit widths excluding clock and reset controls."""
        rtl = _lif_module()
        assert rtl.total_input_bits == 64

    def test_total_output_bits(self) -> None:
        """Sum output bit widths across scalar and vector output ports."""
        rtl = _lif_module()
        assert rtl.total_output_bits == 17

    def test_dense_module(self) -> None:
        """Parse SystemVerilog logic ports from a dense-layer module header."""
        rtl = _dense_module()
        assert rtl.name == "sc_dense_layer_core"
        assert len(rtl.input_ports) == 1
        assert rtl.input_ports[0].name == "input_bus"
        assert rtl.input_ports[0].width == 8

    def test_no_module_raises(self) -> None:
        """Refuse Verilog source text that lacks any module declaration."""
        with pytest.raises(ValueError, match="No module"):
            RTLModule.from_verilog_source("// empty file")

    def test_sv_decl_scalar_vector_and_array(self) -> None:
        """Format SystemVerilog port declarations for vectors and unpacked arrays."""
        vec = ModulePort("foo", PortDirection.INPUT, PortType.LOGIC, 8, False)
        arr = ModulePort("bar", PortDirection.OUTPUT, PortType.WIRE, 4, True, True, 6)
        assert vec.sv_decl == "input logic [7:0] foo"
        assert arr.sv_decl == "output wire signed [3:0] bar [0:5]"

    def test_real_axi_wrapper_parses_parameterised_widths_and_amba_clocks(
        self, tmp_path: Path
    ) -> None:
        """Parse committed AXI wrappers without turning net keywords into port names."""
        source = (REPO_ROOT / "hdl/sc_shd_axi_wrapper.v").read_text(encoding="utf-8")
        rtl = RTLModule.from_verilog_source(source)
        assert rtl.name == "sc_shd_axi_wrapper"
        assert rtl.clock_port is not None and rtl.clock_port.name == "S_AXI_ACLK"
        assert rtl.reset_port is not None and rtl.reset_port.name == "S_AXI_ARESETN"

        ports = {p.name: p for p in rtl.ports}
        assert "wire" not in ports
        assert ports["S_AXI_AWADDR"].width == 8
        assert ports["S_AXI_WDATA"].width == 32
        assert ports["S_AXI_WSTRB"].width == 4
        assert ports["S_AXI_RDATA"].width == 32
        assert ports["S_AXI_BRESP"].width == 2

        bench = UVMGenerator().generate(rtl)
        assert "@(posedge vif.S_AXI_ACLK);" in bench.driver_sv
        assert "@(posedge vif.S_AXI_ACLK);" in bench.monitor_sv
        assert ".C_S_AXI_DATA_WIDTH(32), .C_S_AXI_ADDR_WIDTH(8)" in bench.top_sv
        links = UVMGenerator().generate_formal_links(rtl)
        assert links
        assert all("@(posedge S_AXI_ACLK)" in link.assertion_sv for link in links)

        assert shutil.which("iverilog") is not None
        if_start = bench.top_sv.index("interface sc_shd_axi_wrapper_if")
        if_end = bench.top_sv.index("endinterface") + len("endinterface")
        if_text = bench.top_sv[if_start:if_end]
        harness = (
            f"{if_text}\n"
            "module tb_if_width_check;\n"
            "    logic clk = 1'b0;\n"
            "    sc_shd_axi_wrapper_if intf(.S_AXI_ACLK(clk));\n"
            "    initial begin\n"
            "        if ($bits(intf.S_AXI_AWADDR) !== 8) $fatal(1);\n"
            "        if ($bits(intf.S_AXI_WDATA) !== 32) $fatal(1);\n"
            "        if ($bits(intf.S_AXI_WSTRB) !== 4) $fatal(1);\n"
            "        if ($bits(intf.S_AXI_RDATA) !== 32) $fatal(1);\n"
            "        if ($bits(intf.S_AXI_BRESP) !== 2) $fatal(1);\n"
            '        $display("PASS");\n'
            "        $finish;\n"
            "    end\n"
            "endmodule\n"
        )
        sv_file = tmp_path / "tb_if_width_check.sv"
        vvp_file = tmp_path / "tb_if_width_check.vvp"
        sv_file.write_text(harness, encoding="utf-8")
        comp = subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp_file), str(sv_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert comp.returncode == 0, comp.stderr
        sim = subprocess.run(
            ["vvp", str(vvp_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert sim.returncode == 0 and "PASS" in sim.stdout, sim.stderr

    def test_real_aer_router_and_priority_queue_parse_clog2_ternary_and_2d_arrays(self) -> None:
        """Resolve $clog2, ternary expressions, and 2D unpacked arrays on committed AER HDL."""
        router_src = (REPO_ROOT / "hdl/sc_aer_router.v").read_text(encoding="utf-8")
        router = RTLModule.from_verilog_source(router_src)
        ports = {p.name: p for p in router.ports}
        assert ports["in_neuron_id"].width == 7
        assert ports["fanout_count"].width == 5
        assert ports["fanout_count"].is_array is True
        assert ports["fanout_count"].array_size == 128
        assert ports["target_id"].width == 7
        assert ports["target_id"].is_array is True
        assert ports["target_id"].array_size == 128 * 32
        assert router.total_input_bits > 128 * 32 * 7

        pq_src = (REPO_ROOT / "hdl/sc_aer_priority_queue.v").read_text(encoding="utf-8")
        pq = RTLModule.from_verilog_source(pq_src)
        pq_ports = {p.name: p for p in pq.ports}
        assert pq_ports["occupancy"].width == 5

        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={
                "out_target_id": "7'd0",
                "out_weight": "16'sd0",
                "out_timestamp": "16'd0",
                "out_priority": "1'b0",
            },
        )
        bench = UVMGenerator(scoreboard=sb).generate(router)
        assert "rand logic [4:0] fanout_count [0:127];" in bench.transaction_sv
        assert "txn.fanout_count = '{default: '0};" in bench.sequence_sv
        assert "c_fanout_count_density" not in bench.transaction_sv

    def test_wide_localparam_and_unpacked_output_ports_emit_valid_sv_bounds(self) -> None:
        """Resolve body localparams, non-integer parameters, inline ternaries, and >32-bit bounds."""
        src = """
        /* Header comment with commas (a, b) and module word */
        module wide_array_dut #(
            parameter real GAIN = 0.5,
            parameter integer LANES = 4,
            ,
            localparam int EXTRA = 2
        )(
            input  wire                     custom_aclk, custom_aresetn,
            input  wire [FRAME_W-1:0]       frame_in,
            input  wire [(LANES > 1 ? $clog2(LANES) : 1)-1:0] lane_sel,
            input  wire [3:0]               arr_only [(LANES > 1 ? LANES : 2)],
            output logic [FRAME_W-1:0]      frame_out,
            output logic [7:0]              hist_out [0:LANES+EXTRA-1]
        );
            localparam integer NI = 140;
            localparam integer LANES = 999; // header LANES=4 takes precedence
            localparam GAIN2 = 0.25;
            localparam integer FRAME_W = NI * 4;
        endmodule
        """
        rtl = RTLModule.from_verilog_source(src)
        assert rtl.clock_port is not None and rtl.clock_port.name == "custom_aclk"
        assert rtl.reset_port is not None and rtl.reset_port.name == "custom_aresetn"
        ports = {p.name: p for p in rtl.ports}
        assert ports["frame_in"].width == 560
        assert ports["lane_sel"].width == 2
        assert ports["arr_only"].is_array is True and ports["arr_only"].array_size == 4
        assert ports["hist_out"].is_array is True and ports["hist_out"].array_size == 6
        assert rtl.total_output_bits == 560 + 8 * 6

        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"frame_out": "txn.frame_in"},
        )
        bench = UVMGenerator(scoreboard=sb).generate(rtl)
        assert "560'h" in bench.coverage_sv
        assert "560'h" in bench.bind_sv
        assert "input logic [7:0] hist_out [0:5]" in bench.bind_sv

        arr_only_rtl = RTLModule(
            name="arr_only_mod",
            ports=[
                ModulePort("clk", PortDirection.INPUT),
                ModulePort("rst_n", PortDirection.INPUT),
                ModulePort("vec_in", PortDirection.INPUT, width=4, is_array=True, array_size=2),
                ModulePort("vec_out", PortDirection.OUTPUT, width=4, is_array=True, array_size=2),
            ],
        )
        arr_bench = UVMGenerator().generate(arr_only_rtl)
        assert 'return "arr_only_mod_txn";' in arr_bench.transaction_sv

    def test_constant_expression_operators_bases_and_refusals(self) -> None:
        """Evaluate SystemVerilog integer literals, power, shifts, and fail-closed refusals."""
        syms = {"W": 16, "Z": 0}
        assert _eval_sv_int("+16'h00_10 - 4'b0011 + 8'o10 + 8'sd4", syms) == 16 - 3 + 8 + 4
        assert _eval_sv_int("(W >> 2) << 1", syms) == 8
        assert _eval_sv_int("2 ** 3 ** 2", syms) == 64
        assert _eval_sv_int("-2 ** 2", syms) == 4
        assert _eval_sv_int("-2 ** 3", syms) == -8
        assert _eval_sv_int("9 ** 0 + 0 ** 5 + 1 ** 99 + (-1) ** 3 + (-1) ** 4", syms) == 2
        assert _eval_sv_int("(-7 / 2) * 10 + (-7 % 2)", syms) == -31
        assert _eval_sv_int(f"{(1 << 60) + 3} / 2", syms) == ((1 << 60) + 3) // 2
        assert _eval_sv_int(f"{(1 << 62) + 1} / 1", syms) == (1 << 62) + 1
        assert _eval_sv_int(f"{(1 << 62) + 1} % 3", syms) == ((1 << 62) + 1) % 3
        assert _eval_sv_int("(~0 & 3) ^ (1 | 2)", syms) == 0
        assert _eval_sv_int("(W > 8 && !Z) ? $clog2(1) : $clog2(9)", syms) == 0
        assert _eval_sv_int("(W < 8 || Z != 0) ? 99 : $clog2(9)", syms) == 4
        assert _eval_sv_int("(W <= 16) + (W >= 16) + (W == 16)", syms) == 3

        for bad_expr, pattern in [
            ("   ", "Empty constant expression"),
            ("16 @ 2", "Unsupported token"),
            ("1 +", "Malformed constant expression"),
            ("1 ? 2", "Malformed constant expression"),
            ("1 2", "Trailing tokens"),
            (")", "Unexpected token"),
            ("UNKNOWN + 1", "Unresolved parameter"),
            ("fn_call(4)", "Unsupported constant function"),
            ("8 / 0", "Division by zero"),
            ("8 % 0", "Division by zero"),
            ("$clog2(-1)", "Negative \\$clog2"),
            ("2 ** -1", "Negative exponent"),
            ("2 ** 100", "exceeds 2\\*\\*64"),
            ("3 ** 41", "exceeds 2\\*\\*64"),
            ("1 << 65", "Shift count"),
            ("1 << -1", "Shift count"),
            ("(1 << 64) + 1", "exceeds 2\\*\\*64"),
        ]:
            with pytest.raises(ValueError, match=pattern):
                _eval_sv_int(bad_expr, syms)

    def test_power_grouped_params_unsigned_initialisers_nonansi_and_interface_refusal(
        self,
    ) -> None:
        """Cover power widths, grouped params/localparams, unsigned types, initialisers, and non-ANSI."""
        m_pow = RTLModule.from_verilog_source(
            "module m #(parameter W = 3) (input wire [2**W-1:0] d, output wire q); endmodule"
        )
        assert [(p.name, p.width) for p in m_pow.ports] == [("d", 8), ("q", 1)]

        m_grp = RTLModule.from_verilog_source(
            "module m #(parameter A = 4, B = 8, byte C = 2) "
            "(input wire [B-1:0] d, output wire [C-1:0] q); endmodule"
        )
        assert [(p.name, p.width) for p in m_grp.ports] == [("d", 8), ("q", 2)]
        assert [(p.name, p.value, p.param_type) for p in m_grp.params] == [
            ("A", "4", "int"),
            ("B", "8", "int"),
            ("C", "2", "byte"),
        ]

        m_uns = RTLModule.from_verilog_source(
            "module m #(parameter int unsigned W = 8) "
            "(input wire unsigned [W-1:0] d, output wire q); endmodule"
        )
        assert m_uns.ports[0].width == 8 and m_uns.ports[0].is_signed is False

        m_init = RTLModule.from_verilog_source(
            "module m (input wire clk, output reg [3:0] q = 4'd0, output wire done); endmodule"
        )
        assert [(p.name, p.width) for p in m_init.ports] == [("clk", 1), ("q", 4), ("done", 1)]

        m_nonansi = RTLModule.from_verilog_source(
            "module m (a, b); "
            "function automatic bit f(input logic [15:0] a); f = a[0]; endfunction "
            "input [7:0] a; output b; endmodule"
        )
        assert [(p.name, p.width, p.direction) for p in m_nonansi.ports] == [
            ("a", 8, PortDirection.INPUT),
            ("b", 1, PortDirection.OUTPUT),
        ]

        m_bodylp = RTLModule.from_verilog_source(
            "module m (input wire [N-1:0] d); localparam int unsigned A = 2, N = A*4; endmodule"
        )
        assert [(p.name, p.width) for p in m_bodylp.ports] == [("d", 8)]

        with pytest.raises(ValueError, match="axi_if\\.slave bus"):
            RTLModule.from_verilog_source("module m (axi_if.slave bus, input wire clk); endmodule")
        with pytest.raises(ValueError, match="Unsupported port declaration: 'ghost'"):
            RTLModule.from_verilog_source("module m (a, ghost); input [7:0] a; endmodule")
        with pytest.raises(ValueError, match="Empty port initialiser"):
            RTLModule.from_verilog_source("module m (output reg [3:0] q = ); endmodule")
        with pytest.raises(ValueError, match="Unsupported parameter declaration"):
            RTLModule.from_verilog_source("module m #(bad_param) (input wire clk); endmodule")

    def test_unbalanced_parens_and_invalid_dimensions_fail_closed(self) -> None:
        """Refuse malformed module headers, packed ranges, unpacked bounds, and missing symbols."""
        with pytest.raises(ValueError, match="Unbalanced parentheses"):
            RTLModule.from_verilog_source("module broken #(parameter W = 8 (input clk); endmodule")
        with pytest.raises(ValueError, match="Non-positive unpacked dimension"):
            RTLModule.from_verilog_source("module bad_arr(input logic [7:0] x [0]); endmodule")
        with pytest.raises(ValueError, match="Malformed packed port range"):
            RTLModule.from_verilog_source("module bad_pk(input logic [8] x); endmodule")
        with pytest.raises(ValueError, match="Unresolved parameter"):
            RTLModule.from_verilog_source(
                "module bad_w(input wire [MISSING-1:0] s_tdata); endmodule"
            )
        no_end = RTLModule.from_verilog_source(
            "module open_body(input wire [W-1:0] x); localparam int W = 5;"
        )
        assert no_end.ports[0].width == 5

        with pytest.raises(ValueError, match="exceeds 2\\*\\*64"):
            RTLModule.from_verilog_source(
                "module bad_param_val #(parameter W = 2 ** 100) (input wire [W-1:0] d); endmodule"
            )
        overridden = RTLModule.from_verilog_source(
            "module ok_override #(parameter W = 2 ** 100, W = 8) (input wire [W-1:0] d); endmodule"
        )
        assert overridden.ports[0].width == 8

    def test_conditional_compilation_no_space_signed_and_directive_refusals(self) -> None:
        """Evaluate ifdef/ifndef/elsif/else/endif against defines and refuse other directives."""
        cond_src = """
        `timescale 1ns/1ps
        `default_nettype none
        module cond_dut #(
            parameter TAG = "`ifdef NOT_A_DIRECTIVE",
            localparam signed[63:0] ONE = 64'sd4294967296
        )(
            input wire clk,
            input wire signed[63:0]current_t
        `ifdef OUTER
            `ifdef INNER
            , output wire [7:0] inner_dbg
            `elsif ALT
            , output wire [15:0] alt_dbg
            `else
            , output wire [3:0] outer_else_dbg
            `endif
        `elsif FALLBACK
            , output wire [1:0] fb_dbg
        `else
            `ifndef SUPPRESS_SYN
            , output wire syn_ok
            `endif
        `endif
        );
        endmodule
        `default_nettype wire
        """
        syn = RTLModule.from_verilog_source(cond_src)
        assert [(p.name, p.width, p.is_signed) for p in syn.ports] == [
            ("clk", 1, False),
            ("current_t", 64, True),
            ("syn_ok", 1, False),
        ]
        suppressed = RTLModule.from_verilog_source(cond_src, defines={"SUPPRESS_SYN"})
        assert [p.name for p in suppressed.ports] == ["clk", "current_t"]
        inner = RTLModule.from_verilog_source(cond_src, defines={"OUTER", "INNER"})
        assert [(p.name, p.width) for p in inner.ports] == [
            ("clk", 1),
            ("current_t", 64),
            ("inner_dbg", 8),
        ]
        alt = RTLModule.from_verilog_source(cond_src, defines={"OUTER", "ALT"})
        assert [(p.name, p.width) for p in alt.ports] == [
            ("clk", 1),
            ("current_t", 64),
            ("alt_dbg", 16),
        ]
        outer_else = RTLModule.from_verilog_source(cond_src, defines={"OUTER"})
        assert [(p.name, p.width) for p in outer_else.ports] == [
            ("clk", 1),
            ("current_t", 64),
            ("outer_else_dbg", 4),
        ]
        fb = RTLModule.from_verilog_source(cond_src, defines={"FALLBACK"})
        assert [(p.name, p.width) for p in fb.ports] == [
            ("clk", 1),
            ("current_t", 64),
            ("fb_dbg", 2),
        ]

        top_inc = RTLModule.from_verilog_source(
            '`include "lut.vh"\nmodule m(input wire clk); localparam X = `LUT_0; endmodule'
        )
        assert [p.name for p in top_inc.ports] == ["clk"]

        for bad_src, pattern in [
            ('module m(\n`include "ports.vh"\ninput wire clk); endmodule', "`include"),
            ("module m #(\n`define W 8\nparameter A = 1)(input wire clk); endmodule", "`define"),
            ("module m(input wire [`W-1:0] d); endmodule", "`W"),
            ("module m(a); input [`W-1:0] a; endmodule", "`W"),
            ("module m(input wire [A-1:0] d); localparam A = `W; endmodule", "Unsupported token"),
            ("module m(`ifdef 1\ninput wire clk\n`endif\n); endmodule", "Missing macro name"),
            (
                "module m(`ifdef A\n`elsif 1\n`endif\ninput wire clk); endmodule",
                "Missing macro name after `elsif",
            ),
            ("module m(`elsif A\ninput wire clk); endmodule", "Unexpected `elsif"),
            (
                "module m(`ifdef A\n`else\n`elsif B\n`endif\ninput wire clk); endmodule",
                "Unexpected `elsif",
            ),
            ("module m(`else\ninput wire clk); endmodule", "Unexpected `else"),
            (
                "module m(`ifdef A\n`else\n`else\n`endif\ninput wire clk); endmodule",
                "Unexpected `else",
            ),
            ("module m(`endif\ninput wire clk); endmodule", "Unmatched `endif"),
            ("module m(`ifdef A\ninput wire clk); endmodule", "Unterminated conditional"),
        ]:
            with pytest.raises(ValueError, match=pattern):
                RTLModule.from_verilog_source(bad_src)

    def test_tracked_repository_rtl_corpus_parses_completely(self) -> None:
        """Parse every Git-tracked Verilog and SystemVerilog header in the repository."""
        external_vh_port_modules = {
            "hdl/shd_event/tech/generic/sc_mem.v": "Unsupported constant function 'sc_shape_bits'",
            "hdl/shd_event/tech/xilinx_7series/sc_mem.v": (
                "Unsupported constant function 'sc_shape_bits'"
            ),
        }
        formal_port_modules = {
            "hdl/sc_bitstream_encoder.v",
            "hdl/sc_decorrelator.v",
            "hdl/sc_dense_layer_core.v",
            "hdl/sc_firing_rate_bank.v",
            "hdl/sc_halton16_source.v",
            "hdl/sc_lif_neuron.v",
            "src/sc_neurocore/hdl/primitives/sc_bitstream_encoder.v",
            "src/sc_neurocore/hdl/primitives/sc_dense_layer_core.v",
            "src/sc_neurocore/hdl/primitives/sc_firing_rate_bank.v",
            "src/sc_neurocore/hdl/primitives/sc_lif_neuron.v",
        }
        no_space_signed_modules = {
            "hdl/formal/catalogue/energy_lif.v",
            "hdl/formal/catalogue/energy_lif_formal.v",
            "hdl/formal/catalogue/sc_normalized_energy_lif.v",
            "hdl/formal/catalogue/sc_normalized_energy_lif_formal.v",
            "hdl/formal/catalogue/sc_sigma_delta.v",
            "hdl/formal/catalogue/sc_sigma_delta_accumulator.v",
            "hdl/formal/catalogue/sc_sigma_delta_accumulator_formal.v",
            "hdl/formal/catalogue/sc_sigma_delta_formal.v",
        }

        def check_entry(rel: str, source: str) -> None:
            if rel in external_vh_port_modules:
                with pytest.raises(ValueError, match=external_vh_port_modules[rel]):
                    RTLModule.from_verilog_source(source)
                return
            syn_mod = RTLModule.from_verilog_source(source)
            formal_mod = RTLModule.from_verilog_source(source, defines={"FORMAL"})
            assert syn_mod.name
            assert formal_mod.name == syn_mod.name
            if rel in formal_port_modules:
                assert len(formal_mod.ports) > len(syn_mod.ports)

        check_entry(
            "hdl/shd_event/tech/generic/sc_mem.v",
            "module sc_mem #(parameter SHAPE = 0)(rmask); "
            "localparam BITS = sc_shape_bits(SHAPE); input [BITS-1:0] rmask; endmodule",
        )
        listed = tracked_files(REPO_ROOT, ".")
        assert listed is not None
        rtl_rels = [rel for rel in listed if rel.endswith((".v", ".sv"))]
        seen_rel = set(rtl_rels)
        for rel in rtl_rels:
            check_entry(rel, (REPO_ROOT / rel).read_text(encoding="utf-8"))

        assert formal_port_modules <= seen_rel
        assert no_space_signed_modules <= seen_rel
