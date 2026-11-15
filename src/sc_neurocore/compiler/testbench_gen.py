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
from .equation_lowering import prepare_neuron
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

    Drives the module with constant current for ``n_steps`` logical steps and monitors
    spike_out and state outputs. Run with the plusarg ``+vcd`` (``vvp sim +vcd``), it also
    writes the waveform ``tb_<module_name>.vcd`` in its working directory; otherwise no file.

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
    fixed_point = Q88(data_width=data_width, fraction=fraction)
    current_literal = fixed_point.encode_signed_literal(input_current)
    state_vars = list(prepare_neuron(neuron).equations)
    q_format = f"Q{data_width - fraction}.{fraction}"
    lines = _module_header(module_name, state_vars, data_width, current_literal)
    lines += _stimulus_block(module_name, state_vars, q_format, n_steps, cycles_per_step)
    return "\n".join(lines)


def _module_header(
    module_name: str, state_vars: list[str], data_width: int, current_literal: str
) -> list[str]:
    """Return the testbench's declarations, the instance of the module and its clock.

    Parameters
    ----------
    module_name : str
        The module under test.
    state_vars : list of str
        The state variables, each with an ``<name>_out`` port.
    data_width : int
        Bit width of the state outputs.
    current_literal : str
        The Verilog literal of the constant input current.

    Returns
    -------
    list of str
        The source lines up to and including the spike counter's declaration.
    """
    ports = ["    .clk(clk),", "    .rst_n(rst_n),", f"    .I_t({current_literal}),"]
    ports.append("    .spike_out(spike_out),")
    ports.extend(f"    .{var}_out({var}_out)," for var in state_vars)
    ports[-1] = ports[-1].rstrip(",")
    lines = [f"// Auto-generated testbench for {module_name}", "// SC-NeuroCore equation compiler"]
    lines += ["`timescale 1ns / 1ps", "", f"module tb_{module_name};", ""]
    lines += ["reg clk;", "reg rst_n;", "wire spike_out;"]
    lines += [f"wire signed [{data_width - 1}:0] {var}_out;" for var in state_vars]
    lines += ["", f"{module_name} uut (", *ports, ");", ""]
    lines += ["// Clock: 10ns period (100 MHz)", "initial clk = 0;", "always #5 clk = ~clk;", ""]
    return [*lines, "integer spike_count;", ""]


def _stimulus_block(
    module_name: str, state_vars: list[str], q_format: str, n_steps: int, cycles_per_step: int
) -> list[str]:
    """Return the reset, the run, the reports and the end of the testbench.

    The waveform dump is guarded by the plusarg ``+vcd``.

    Parameters
    ----------
    module_name : str
        The module under test.
    state_vars : list of str
        The state variables whose final values are reported.
    q_format : str
        The state outputs' fixed-point format, such as ``Q8.8``, for the reports.
    n_steps : int
        Logical steps to run.
    cycles_per_step : int
        Clock cycles per logical step.

    Returns
    -------
    list of str
        The source lines from ``initial begin`` to ``endmodule``.
    """
    total_cycles = n_steps * cycles_per_step
    lines = ["initial begin", '    if ($test$plusargs("vcd")) begin']
    lines += [
        f'        $dumpfile("tb_{module_name}.vcd");',
        f"        $dumpvars(0, tb_{module_name});",
    ]
    lines += ["    end", "    spike_count = 0;", "", "    // Reset", "    rst_n = 0;", "    #20;"]
    lines += ["    rst_n = 1;", "    @(posedge clk);  // 1 settling cycle after reset", ""]
    lines.append(f"    // Run {n_steps} logical step(s) × {cycles_per_step} cycle(s)/step")
    lines += [f"    repeat ({total_cycles}) begin", "        @(posedge clk);"]
    lines += ["        #1;  // let combinational outputs settle"]
    lines += ["        if (spike_out) spike_count = spike_count + 1;", "    end", ""]
    lines.append(
        f'    $display("Simulation complete: %0d spikes in {total_cycles} cycles", spike_count);'
    )
    lines += [f'    $display("Final {var} = %0d ({q_format})", {var}_out);' for var in state_vars]
    return [*lines, "    $finish;", "end", "", "endmodule"]
