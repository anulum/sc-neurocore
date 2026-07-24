# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_operator_abstraction.py

from __future__ import annotations

"""Tests for abstracting an internal result to a free input port.

Pure text-transformation tests — no formal toolchain required. The end-to-end
unbounded equivalence proof that consumes the abstraction lives with the
equivalence runner tests and self-skips without ``sby``.
"""
import pytest
from sc_neurocore.compiler.equivalence_miter import parse_module_interface
from sc_neurocore.compiler.operator_abstraction import LiftedSignal, abstract_to_free_inputs

_MODULE = """`timescale 1ns/1ps
module foo #(parameter integer W = 8)(
    input wire clk,
    input wire signed [W-1:0] a,
    input wire signed [W-1:0] b,
    output reg signed [W-1:0] y
);
    wire signed [2*W-1:0] prod;
    wire signed [W-1:0] scaled;
    assign prod = a * b;
    assign scaled = prod >>> 2;
    always @(posedge clk) y <= scaled;
endmodule
"""
_MODULE_INLINE = """`timescale 1ns/1ps
module baz #(parameter integer W = 8)(
    input wire clk,
    input wire signed [W-1:0] a,
    output reg signed [W-1:0] y
);
    wire signed [2*W-1:0] sq = a * a;
    wire signed [W-1:0] scaled = sq >>> 2;
    always @(posedge clk) y <= scaled;
endmodule
"""

__all__ = [
    "pytest",
    "parse_module_interface",
    "LiftedSignal",
    "abstract_to_free_inputs",
    "_MODULE",
    "_MODULE_INLINE",
]
