# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_equivalence_miter.py

from __future__ import annotations

"""Unit tests for sequential-equivalence miter construction (no external tools)."""
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

__all__ = ['pytest', 'MiterPort', '_eval_width_expr', 'build_equivalence_miter', 'parse_module_interface', '_LIF_REFERENCE', '_lif_ports']
