# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog testbench generation

"""Generate Verilog testbenches for compiled equation neurons."""

from __future__ import annotations

from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler_config import Q88


def generate_testbench(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    n_steps: int = 200,
    input_current: float = 1.0,
    data_width: int = 16,
    fraction: int = 8,
    cycles_per_step: int = 1,
) -> str:
    """Generate a Verilog testbench for a compiled equation neuron.

    Drives the module with constant current for ``n_steps`` logical steps and
    monitors spike_out and state outputs, producing a VCD waveform.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron (same one passed to compile_to_verilog).
    module_name : str
        Must match the module name used in compile_to_verilog.
    n_steps : int
        Number of logical integration steps to drive.
    input_current : float
        Constant input current (Q-encoded internally).
    data_width : int
        Bit width matching the compiled module.
    fraction : int
        Fractional bits matching the compiled module.
    cycles_per_step : int
        Clock cycles per logical step. Combinational modules advance one logical
        step per clock (``1``, the default). A pipelined module advances one
        logical step every ``latency + 1`` clocks and gates ``spike_out`` to pulse
        only on that valid cycle, so pass ``latency + 1`` here to drive ``n_steps``
        logical steps; the spike count over the run is unchanged by the padding.

    Returns
    -------
    str
        Verilog testbench source code.
    """
    if cycles_per_step < 1:
        raise ValueError(f"cycles_per_step must be >= 1, got {cycles_per_step}")
    total_cycles = n_steps * cycles_per_step
    q = Q88(data_width=data_width, fraction=fraction)
    i_val = q.encode_signed_literal(input_current)

    state_vars = list(neuron.equations.keys())
    port_connections = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        f"    .I_t({i_val}),",
        "    .spike_out(spike_out),",
    ]
    wire_decls = []
    for var in state_vars:
        port_connections.append(f"    .{var}_out({var}_out),")
        wire_decls.append(f"wire signed [{data_width - 1}:0] {var}_out;")
    port_connections[-1] = port_connections[-1].rstrip(",")

    lines = [
        f"// Auto-generated testbench for {module_name}",
        "// SC-NeuroCore equation compiler",
        "`timescale 1ns / 1ps",
        "",
        f"module tb_{module_name};",
        "",
        "reg clk;",
        "reg rst_n;",
        "wire spike_out;",
    ]
    lines.extend(wire_decls)
    lines.append("")
    lines.append(f"{module_name} uut (")
    lines.extend(port_connections)
    lines.append(");")
    lines.append("")
    lines.append("// Clock: 10ns period (100 MHz)")
    lines.append("initial clk = 0;")
    lines.append("always #5 clk = ~clk;")
    lines.append("")
    lines.append("integer spike_count;")
    lines.append("")
    lines.append("initial begin")
    lines.append(f'    $dumpfile("tb_{module_name}.vcd");')
    lines.append(f"    $dumpvars(0, tb_{module_name});")
    lines.append("    spike_count = 0;")
    lines.append("")
    lines.append("    // Reset")
    lines.append("    rst_n = 0;")
    # 20ns reset
    lines.append("    #20;")
    lines.append("    rst_n = 1;")
    lines.append("    @(posedge clk);  // 1 settling cycle after reset")
    lines.append("")
    lines.append(f"    // Run {n_steps} logical step(s) × {cycles_per_step} cycle(s)/step")
    lines.append(f"    repeat ({total_cycles}) begin")
    lines.append("        @(posedge clk);")
    lines.append("        #1;  // let combinational outputs settle")
    lines.append("        if (spike_out) spike_count = spike_count + 1;")
    lines.append("    end")
    lines.append("")
    lines.append(
        f'    $display("Simulation complete: %0d spikes in {total_cycles} cycles", spike_count);'
    )
    for var in state_vars:
        lines.append(
            f'    $display("Final {var} = %0d (Q{data_width - fraction}.{fraction})", {var}_out);'
        )
    lines.append("    $finish;")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
