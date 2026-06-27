# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM generator package API tests

"""Package-facade tests for the UVM generator public API."""

from __future__ import annotations

import sc_neurocore.uvm_gen as uvm_gen
from sc_neurocore.uvm_gen.uvm_gen import (
    CoverageSpec,
    FormalLink,
    ModuleParam,
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    SIM_TARGETS,
    ScoreboardConfig,
    SimTarget,
    StimulusConfig,
    UVMBenchmark,
    UVMGenerator,
)


def test_uvm_gen_package_exports_generator_api() -> None:
    """Package entry point re-exports the documented UVM generator API."""
    expected = [
        "CoverageSpec",
        "FormalLink",
        "ModuleParam",
        "ModulePort",
        "PortDirection",
        "PortType",
        "RTLModule",
        "SIM_TARGETS",
        "ScoreboardConfig",
        "SimTarget",
        "StimulusConfig",
        "UVMBenchmark",
        "UVMGenerator",
    ]

    assert uvm_gen.__tier__ == "industrial"
    assert uvm_gen.__all__ == expected
    assert uvm_gen.CoverageSpec is CoverageSpec
    assert uvm_gen.FormalLink is FormalLink
    assert uvm_gen.ModuleParam is ModuleParam
    assert uvm_gen.ModulePort is ModulePort
    assert uvm_gen.PortDirection is PortDirection
    assert uvm_gen.PortType is PortType
    assert uvm_gen.RTLModule is RTLModule
    assert uvm_gen.SIM_TARGETS is SIM_TARGETS
    assert uvm_gen.ScoreboardConfig is ScoreboardConfig
    assert uvm_gen.SimTarget is SimTarget
    assert uvm_gen.StimulusConfig is StimulusConfig
    assert uvm_gen.UVMBenchmark is UVMBenchmark
    assert uvm_gen.UVMGenerator is UVMGenerator


def test_uvm_gen_package_import_generates_benchmark() -> None:
    """The package-level UVMGenerator creates a benchmark from package RTLModule."""
    source = """\
module passthrough(
    input logic clk,
    input logic rst_n,
    input logic [7:0] data_i,
    output logic [7:0] data_o
);
endmodule
"""
    rtl = uvm_gen.RTLModule.from_verilog_source(source)
    benchmark = uvm_gen.UVMGenerator().generate(rtl)

    assert benchmark.module_name == "passthrough"
    assert "passthrough_transaction" in benchmark.transaction_sv
    assert "tb_passthrough_top" in benchmark.top_sv
