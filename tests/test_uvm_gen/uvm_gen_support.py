# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_uvm_gen.py

from __future__ import annotations

import pytest
from sc_neurocore.uvm_gen.uvm_gen import (
    CoverageSpec,
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    ScoreboardConfig,
    StimulusConfig,
    SIM_TARGETS,
    UVMBenchmark,
    UVMGenerator,
)

LIF_VERILOG = """\
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
DENSE_VERILOG = """\
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


def lif_module() -> RTLModule:
    return RTLModule.from_verilog_source(LIF_VERILOG)


def dense_module() -> RTLModule:
    return RTLModule.from_verilog_source(DENSE_VERILOG)


PARAMLESS_VERILOG_WITH_BLANK_PORT = """\
module sc_paramless (
    input  wire clk,
    ,
    output wire done
);
endmodule
"""

__all__ = [
    "pytest",
    "CoverageSpec",
    "ModulePort",
    "PortDirection",
    "PortType",
    "RTLModule",
    "ScoreboardConfig",
    "StimulusConfig",
    "SIM_TARGETS",
    "UVMBenchmark",
    "UVMGenerator",
    "LIF_VERILOG",
    "DENSE_VERILOG",
    "lif_module",
    "dense_module",
    "PARAMLESS_VERILOG_WITH_BLANK_PORT",
    "__all__",
]
