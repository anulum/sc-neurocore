# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_formal_property_check.py

from __future__ import annotations

"""Tests for the SymbiYosys RTL-property runner.

Pure tests drive the ``.sby`` generation and the verdict-to-result mapping with a
fake run, so every branch runs without the toolchain (as on CI). The end-to-end
proofs run real ``sby`` tasks and self-skip when the formal toolchain is absent.
"""
from pathlib import Path
import pytest
from sc_neurocore.compiler import formal_property_check
from sc_neurocore.compiler._sby_runner import SbyRun
from sc_neurocore.compiler.formal_property_check import (
    PropertyProofResult,
    formal_tools_available,
    prove_property,
)
_HAS_FORMAL = formal_tools_available()
_needs_formal = pytest.mark.skipif(
    not _HAS_FORMAL, reason="SymbiYosys / Yosys / solver not available"
)
_CAP_RTL = """`timescale 1ns/1ps
module cap(input wire clk, output wire [3:0] c_o);
    reg [3:0] c = 4'd0;
    always @(posedge clk) c <= c + 4'd1;
    assign c_o = c;
`ifdef FORMAL
    cap_sva sva_i (.clk(clk), .c(c));
`endif
endmodule
"""
_CAP_SVA = """`timescale 1ns/1ps
module cap_sva(input logic clk, input logic [3:0] c);
    always @(posedge clk) assert (c <= 4'd15);
endmodule
"""
_BAD_RTL = _CAP_RTL.replace("cap", "bad")
_BAD_SVA = """`timescale 1ns/1ps
module bad_sva(input logic clk, input logic [3:0] c);
    always @(posedge clk) assert (c < 4'd5);
endmodule
"""
_ACC_RTL = """`timescale 1ns/1ps
module acc(input wire clk, input wire [9:0] step, output wire [9:0] acc_o, output wire [4:0] sc_o);
    reg [9:0] acc = 10'd0;
    reg [4:0] sc = 5'd0;
    always @(posedge clk) if (sc < 5'd16) begin acc <= acc + step; sc <= sc + 5'd1; end
    assign acc_o = acc;
    assign sc_o = sc;
`ifdef FORMAL
    acc_sva sva_i (.clk(clk), .step(step), .acc(acc), .sc(sc));
`endif
endmodule
"""
_ACC_SVA_STRONG = """`timescale 1ns/1ps
module acc_sva(input logic clk, input logic [9:0] step, input logic [9:0] acc, input logic [4:0] sc);
    always @(posedge clk) begin
        assume (step <= 10'd8);
        assert (acc <= sc * 8);
        assert (sc <= 5'd16);
        assert (acc <= 10'd128);
    end
endmodule
"""
_ACC_SVA_WEAK = """`timescale 1ns/1ps
module acc_sva(input logic clk, input logic [9:0] step, input logic [9:0] acc, input logic [4:0] sc);
    always @(posedge clk) begin
        assume (step <= 10'd8);
        assert (acc <= 10'd128);
    end
endmodule
"""
def _fake_run(**fields: object) -> SbyRun:
    base = {"verdict": "PASS", "rc": 0, "returncode": 0}
    base.update(fields)
    return SbyRun(**base)  # type: ignore[arg-type]

__all__ = ['Path', 'pytest', 'formal_property_check', 'SbyRun', 'PropertyProofResult', 'formal_tools_available', 'prove_property', '_HAS_FORMAL', '_needs_formal', '_CAP_RTL', '_CAP_SVA', '_BAD_RTL', '_BAD_SVA', '_ACC_RTL', '_ACC_SVA_STRONG', '_ACC_SVA_WEAK', '_fake_run']
