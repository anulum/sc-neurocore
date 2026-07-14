# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM top-level, formal, simulator, and regression harness emitters

"""Emit deterministic top-level, formal, simulator, and regression artifacts."""

from __future__ import annotations

import textwrap
from typing import List

from sc_neurocore.uvm_gen._config import FormalLink, SIM_TARGETS, SPDX_HEADER
from sc_neurocore.uvm_gen._rtl import RTLModule


def _emit_top(rtl: RTLModule) -> str:
    module_name = rtl.name
    clock = rtl.clock_port
    reset = rtl.reset_port
    clock_name = clock.name if clock else "clk"
    reset_name = reset.name if reset else "rst_n"

    interface_signals = []
    for port in rtl.ports:
        if not port.is_clock and not port.is_reset:
            signed = " signed" if port.is_signed else ""
            width = f" [{port.width - 1}:0]" if port.width > 1 else ""
            interface_signals.append(f"    logic{signed}{width} {port.name};")
    interface_block = "\n".join(interface_signals) if interface_signals else "    logic [7:0] data;"

    dut_connections = []
    for port in rtl.ports:
        dut_connections.append(
            f"        .{port.name}({'intf.' + port.name if not port.is_clock and not port.is_reset else port.name})"
        )
    dut_block = ",\n".join(dut_connections)

    parameter_override = ""
    if rtl.params:
        parameter_values = ", ".join(f".{param.name}({param.value})" for param in rtl.params)
        parameter_override = f" #({parameter_values})"

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Top-Level Testbench for {module_name}

`include "uvm_macros.svh"
import uvm_pkg::*;

interface {module_name}_if(input logic {clock_name});
{interface_block}
endinterface

module tb_{module_name}_top;
    logic {clock_name} = 0;
    logic {reset_name} = 0;

    always #5 {clock_name} = ~{clock_name};

    {module_name}_if intf(.{clock_name}({clock_name}));

    {module_name}{parameter_override} dut (
{dut_block}
    );

    initial begin
        uvm_config_db #(virtual {module_name}_if)::set(null, "*", "vif", intf);
        {reset_name} = 0;
        #20;
        {reset_name} = 1;
    end

    class {module_name}_test extends uvm_test;
        `uvm_component_utils({module_name}_test)
        {module_name}_env env;

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            env = {module_name}_env::type_id::create("env", this);
        endfunction

        task run_phase(uvm_phase phase);
            {module_name}_random_seq seq;
            phase.raise_objection(this);
            seq = {module_name}_random_seq::type_id::create("seq");
            seq.start(env.agt.sqr);
            phase.drop_objection(this);
        endtask
    endclass

    initial begin
        run_test("{module_name}_test");
    end
endmodule
""")


def _emit_sby(rtl: RTLModule) -> str:
    module_name = rtl.name
    return textwrap.dedent(f"""\
[options]
mode prove
depth 25

[engines]
smtbmc

[script]
read -formal tb_{module_name}_top.sv
read -formal ../{module_name}.v
prep -top tb_{module_name}_top

[files]
tb_{module_name}_top.sv
../{module_name}.v
""")


def _filelist(rtl: RTLModule) -> List[str]:
    module_name = rtl.name
    return [
        f"{module_name}_transaction.sv",
        f"{module_name}_sequence.sv",
        f"{module_name}_driver.sv",
        f"{module_name}_monitor.sv",
        f"{module_name}_scoreboard.sv",
        f"{module_name}_coverage.sv",
        f"{module_name}_agent.sv",
        f"{module_name}_env.sv",
        f"tb_{module_name}_top.sv",
        f"{module_name}_verify.sby",
        f"{module_name}_bind.sv",
    ]


def _emit_bind(rtl: RTLModule) -> str:
    module_name = rtl.name
    reset = rtl.reset_port
    clock = rtl.clock_port
    reset_name = reset.name if reset else "rst_n"
    clock_name = clock.name if clock else "clk"

    assertions = []
    for port in rtl.output_ports:
        if port.width == 1:
            assertions.append(
                f"    // Reset assertion for {port.name}\n"
                f"    property p_{port.name}_resets;\n"
                f"        @(posedge {clock_name}) !{reset_name} |-> {port.name} == 0;\n"
                f"    endproperty\n"
                f"    a_{port.name}_rst: assert property(p_{port.name}_resets);\n"
                f"    c_{port.name}_active: cover property(\n"
                f"        @(posedge {clock_name}) {reset_name} |-> {port.name} == 1);"
            )
        elif port.width > 1:
            assertions.append(
                f"    // Bounded output for {port.name}\n"
                f"    a_{port.name}_bounded: assert property(\n"
                f"        @(posedge {clock_name}) {reset_name} |-> ({port.name} <= {(1 << port.width) - 1}));\n"
                f"    c_{port.name}_nonzero: cover property(\n"
                f"        @(posedge {clock_name}) {reset_name} |-> ({port.name} != 0));"
            )

    assertion_block = "\n\n".join(assertions) if assertions else "    // No assertions generated"

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Assertion Bind Module for {module_name}

module {module_name}_assertions (
    input logic {clock_name},
    input logic {reset_name},
{chr(10).join(f"    input logic [{port.width - 1}:0] {port.name}," if port.width > 1 else f"    input logic {port.name}," for port in rtl.output_ports).rstrip(",")}
);

{assertion_block}

endmodule

bind {module_name} {module_name}_assertions {module_name}_assert_inst (
    .{clock_name}({clock_name}),
    .{reset_name}({reset_name}),
{chr(10).join(f"    .{port.name}({port.name})," for port in rtl.output_ports).rstrip(",")}
);
""")


def _emit_makefile(rtl: RTLModule, sim: str = "vcs") -> str:
    module_name = rtl.name
    target = SIM_TARGETS.get(sim, SIM_TARGETS["vcs"])
    filelist = f"{module_name}.f"
    compile_command = target.compile_cmd.format(flist=filelist, module=module_name)
    run_command = target.run_cmd.format(test=f"{module_name}_test", module=module_name)
    coverage_command = target.coverage_cmd

    return textwrap.dedent(f"""\
# SC-NeuroCore UVM — Makefile for {module_name} ({target.name})
# Auto-generated by UVM Generator

MODULE = {module_name}
FLIST  = {filelist}

.PHONY: compile sim coverage clean regression

compile:
\t{compile_command}

sim: compile
\t{run_command}

coverage:
\t{coverage_command}

clean:
\trm -rf simv simv.daidir csrc *.vdb *.log work *.ucdb

regression:
\t@echo "Running regression..."
\t$(MAKE) sim
\t$(MAKE) coverage
\t@echo "Regression complete."
""")


def _emit_regression_list(rtl: RTLModule) -> str:
    module_name = rtl.name
    lines = [
        f"# SC-NeuroCore UVM — Regression test list for {module_name}",
        "# Auto-generated by UVM Generator",
        "",
        "# test_name : sequence : iterations",
        f"{module_name}_random   : {module_name}_random_seq  : 1000",
        f"{module_name}_corner   : {module_name}_corner_seq  : 1",
        f"{module_name}_lfsr     : {module_name}_lfsr_seq    : 256",
    ]
    return "\n".join(lines) + "\n"


def _generate_formal_links(rtl: RTLModule) -> List[FormalLink]:
    links = []
    reset = rtl.reset_port
    reset_name = reset.name if reset else "rst_n"

    for port in rtl.output_ports:
        if port.width == 1:
            links.append(
                FormalLink(
                    property_name=f"{port.name}_reset_check",
                    sby_module=f"{rtl.name}_formal",
                    assertion_sv=(
                        f"property p_{port.name}_rst;\n"
                        f"    @(posedge clk) !{reset_name} |-> {port.name} == 0;\n"
                        f"endproperty\n"
                        f"assert property(p_{port.name}_rst);"
                    ),
                    cover_sv=(f"cover property(@(posedge clk) {reset_name} |-> {port.name} == 1);"),
                )
            )

    return links
