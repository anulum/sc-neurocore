# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_equivalence_check.py

from __future__ import annotations

"""Tests for the SymbiYosys equivalence runner.

The proof tests skip when ``sby`` / ``yosys`` are absent (as on CI without the
formal toolchain), mirroring the co-simulation tests' toolchain guard.
"""
from pathlib import Path
import pytest
from sc_neurocore.compiler import _sby_runner, equivalence_check
from sc_neurocore.compiler.equivalence_check import (
    EquivalenceResult,
    formal_tools_available,
    prove_equivalence,
)
from sc_neurocore.compiler.equivalence_miter import MiterPort

_HAS_FORMAL = formal_tools_available()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TINY_DUT = """
module tiny_dut #(parameter integer W = 8)(
    input wire clk,
    input wire rst_n,
    input wire [W-1:0] a,
    input wire [W-1:0] b,
    output reg [W-1:0] y
);
    always @(posedge clk or negedge rst_n)
        if (!rst_n) y <= 0; else y <= a + b;
endmodule
"""
_TINY_REF = """
module tiny_ref #(parameter integer W = 8)(
    input wire clk,
    input wire rst_n,
    input wire [W-1:0] a,
    input wire [W-1:0] b,
    output reg [W-1:0] y
);
    wire [W-1:0] s = b + a;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) y <= 0; else y <= s;
endmodule
"""
_TINY_REF_BAD = _TINY_REF.replace("wire [W-1:0] s = b + a;", "wire [W-1:0] s = b + a + 1;")
_TINY_PORTS = [
    MiterPort("clk", 1, False, "input"),
    MiterPort("rst_n", 1, False, "input"),
    MiterPort("a", 8, False, "input"),
    MiterPort("b", 8, False, "input"),
    MiterPort("y", 8, False, "output"),
]
_needs_formal = pytest.mark.skipif(not _HAS_FORMAL, reason="SymbiYosys / Yosys not available")
_QIF_REF = """`timescale 1ns/1ps
module sc_qif_reference #(
    parameter integer DATA_WIDTH = 16, parameter integer K_SHIFT = 6,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = 1024,
    parameter signed [DATA_WIDTH-1:0] V_RESET = -1024,
    parameter signed [DATA_WIDTH-1:0] V_MIN = -2048
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v;
    wire signed [2*DATA_WIDTH-1:0] v_sq = v * v;
    wire signed [DATA_WIDTH-1:0] dv = v_sq >>> K_SHIFT;
    wire signed [DATA_WIDTH-1:0] v_tmp = v + dv + I_t;
    wire signed [DATA_WIDTH-1:0] v_clamped = (v_tmp < V_MIN) ? V_MIN : v_tmp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v <= 0; v_out <= 0; spike_out <= 1'b0; end
        else if (v_clamped >= V_THRESHOLD) begin spike_out <= 1'b1; v <= V_RESET; v_out <= V_RESET; end
        else begin spike_out <= 1'b0; v <= v_clamped; v_out <= v_clamped; end
    end
endmodule
"""
_QIF_DUT = """`timescale 1ns/1ps
module sc_qif_dut #(
    parameter integer DATA_WIDTH = 16, parameter integer K_SHIFT = 6,
    parameter signed [DATA_WIDTH-1:0] V_THRESHOLD = 1024,
    parameter signed [DATA_WIDTH-1:0] V_RESET = -1024,
    parameter signed [DATA_WIDTH-1:0] V_MIN = -2048
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v_reg;
    wire signed [2*DATA_WIDTH-1:0] v_squared = v_reg * v_reg;
    wire signed [DATA_WIDTH-1:0] quad_term = v_squared >>> K_SHIFT;
    wire signed [DATA_WIDTH-1:0] v_int = (v_reg + quad_term) + I_t;
    wire signed [DATA_WIDTH-1:0] v_floor = (v_int >= V_MIN) ? v_int : V_MIN;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v_reg <= 0; v_out <= 0; spike_out <= 1'b0; end
        else if (v_floor >= V_THRESHOLD) begin spike_out <= 1'b1; v_reg <= V_RESET; v_out <= V_RESET; end
        else begin spike_out <= 1'b0; v_reg <= v_floor; v_out <= v_floor; end
    end
endmodule
"""
_IZH_REF = """`timescale 1ns/1ps
module sc_izh_reference #(
    parameter integer DATA_WIDTH = 16, parameter integer KQ_SHIFT = 6, parameter integer KA_SHIFT = 5,
    parameter signed [DATA_WIDTH-1:0] VR = -60,
    parameter signed [DATA_WIDTH-1:0] VT = -40,
    parameter signed [DATA_WIDTH-1:0] VPEAK = 30,
    parameter signed [DATA_WIDTH-1:0] C_RESET = -50,
    parameter signed [DATA_WIDTH-1:0] D_STEP = 6
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v;
    reg signed [DATA_WIDTH-1:0] u;
    wire signed [DATA_WIDTH-1:0] v_mvr = v - VR;
    wire signed [DATA_WIDTH-1:0] v_mvt = v - VT;
    wire signed [2*DATA_WIDTH-1:0] q_prod = v_mvr * v_mvt;
    wire signed [DATA_WIDTH-1:0] q_scaled = q_prod >>> KQ_SHIFT;
    wire signed [DATA_WIDTH-1:0] dv = q_scaled - u + I_t;
    wire signed [DATA_WIDTH-1:0] v_next = v + dv;
    wire signed [DATA_WIDTH-1:0] bvr = v_mvr <<< 1;
    wire signed [DATA_WIDTH-1:0] du = (bvr - u) >>> KA_SHIFT;
    wire signed [DATA_WIDTH-1:0] u_next = u + du;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v <= VR; u <= 0; v_out <= VR; spike_out <= 1'b0; end
        else if (v_next >= VPEAK) begin spike_out <= 1'b1; v <= C_RESET; v_out <= C_RESET; u <= u_next + D_STEP; end
        else begin spike_out <= 1'b0; v <= v_next; v_out <= v_next; u <= u_next; end
    end
endmodule
"""
_IZH_DUT = """`timescale 1ns/1ps
module sc_izh_dut #(
    parameter integer DATA_WIDTH = 16, parameter integer KQ_SHIFT = 6, parameter integer KA_SHIFT = 5,
    parameter signed [DATA_WIDTH-1:0] VR = -60,
    parameter signed [DATA_WIDTH-1:0] VT = -40,
    parameter signed [DATA_WIDTH-1:0] VPEAK = 30,
    parameter signed [DATA_WIDTH-1:0] C_RESET = -50,
    parameter signed [DATA_WIDTH-1:0] D_STEP = 6
)(
    input wire clk, input wire rst_n, input wire signed [DATA_WIDTH-1:0] I_t,
    output reg spike_out, output reg signed [DATA_WIDTH-1:0] v_out
);
    reg signed [DATA_WIDTH-1:0] v_reg;
    reg signed [DATA_WIDTH-1:0] u_reg;
    wire signed [DATA_WIDTH-1:0] vr_diff = v_reg - VR;
    wire signed [DATA_WIDTH-1:0] vt_diff = v_reg - VT;
    wire signed [2*DATA_WIDTH-1:0] p_prod = vt_diff * vr_diff;
    wire signed [DATA_WIDTH-1:0] p_scaled = p_prod >>> KQ_SHIFT;
    wire signed [DATA_WIDTH-1:0] v_adv = (v_reg + p_scaled) - u_reg + I_t;
    wire signed [DATA_WIDTH-1:0] recover = ((vr_diff + vr_diff) - u_reg) >>> KA_SHIFT;
    wire signed [DATA_WIDTH-1:0] u_adv = u_reg + recover;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v_reg <= VR; u_reg <= 0; v_out <= VR; spike_out <= 1'b0; end
        else if (!(v_adv < VPEAK)) begin spike_out <= 1'b1; v_reg <= C_RESET; v_out <= C_RESET; u_reg <= u_adv + D_STEP; end
        else begin spike_out <= 1'b0; v_reg <= v_adv; v_out <= v_adv; u_reg <= u_adv; end
    end
endmodule
"""

__all__ = [
    "Path",
    "pytest",
    "_sby_runner",
    "equivalence_check",
    "EquivalenceResult",
    "formal_tools_available",
    "prove_equivalence",
    "MiterPort",
    "_HAS_FORMAL",
    "_REPO_ROOT",
    "_TINY_DUT",
    "_TINY_REF",
    "_TINY_REF_BAD",
    "_TINY_PORTS",
    "_needs_formal",
    "_QIF_REF",
    "_QIF_DUT",
    "_IZH_REF",
    "_IZH_DUT",
]
