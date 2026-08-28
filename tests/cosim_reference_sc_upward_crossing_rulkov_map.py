# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained upward-crossing Rulkov RTL reference

"""Execute the retained Rulkov schema's generated Q16.16 RTL."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def sc_upward_crossing_rulkov_q1616_trace(
    n_steps: int, current: float
) -> list[tuple[int, float, float]]:
    """Return event and committed-state rows from generated Q16.16 RTL."""
    neuron = UniversalNeuron.from_schema("sc_upward_crossing_rulkov_map")
    module_name = "sc_upward_crossing_rulkov_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_upward_crossing_rulkov_map_q1616_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] x_out;",
            "wire signed [31:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .x_out(x_out), .y_out(y_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("SC_RULKOV_TRACE %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        rtl_path = root / f"{module_name}.v"
        testbench_path = root / f"tb_{module_name}.v"
        output_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        testbench_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(output_path), str(rtl_path), str(testbench_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(output_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    scale = float(1 << 16)
    rows = re.findall(r"^SC_RULKOV_TRACE (-?\d+) (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    trace = [(int(event), int(x_q) / scale, int(y_q) / scale) for event, x_q, y_q in rows]
    assert len(trace) == n_steps
    return trace
