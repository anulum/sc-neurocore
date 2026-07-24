# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_whitebox_taps.py

from __future__ import annotations

"""Tests for exposing internal signals as observation output ports.

These are pure text-transformation tests — no formal toolchain required. The
end-to-end k-induction proof that consumes the taps lives with the equivalence
runner tests and self-skips without ``sby``.
"""
import pytest
from sc_neurocore.compiler.equivalence_miter import parse_module_interface
from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

_MODULE = """`timescale 1ns/1ps
module foo #(parameter integer W = 8)(
    input wire clk,
    input wire [W-1:0] x,
    output reg [W-1:0] y
);
    reg [W-1:0] s;
    always @(posedge clk) begin s <= x; y <= s; end
endmodule
"""

__all__ = ["pytest", "parse_module_interface", "StateTap", "expose_state_taps", "_MODULE"]
