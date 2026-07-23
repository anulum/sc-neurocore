# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka map co-simulation reference

"""Real emitted-RTL trace contract for the Ibarz-Tanaka map."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def _ibarz_tanaka_verilog_q1616_trace(
    n_steps: int, current: float
) -> list[tuple[int, float, float]]:
    """Return the emitted Ibarz-Tanaka RTL's committed Q16.16 trace."""
    neuron = UniversalNeuron.from_schema("ibarz_tanaka_map")
    module_name = "sc_ibarz_tanaka_rulkov_map"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] v_out;",
            "wire signed [31:0] u_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .v_out(v_out), .u_out(u_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("IBARZ_TRACE %0d %0d %0d", spike_out, uut.v_reg, uut.u_reg);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        tb_path = root / f"tb_{module_name}.v"
        out_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

    scale = float(1 << 16)
    rows = re.findall(r"^IBARZ_TRACE (-?\d+) (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    trace = [(int(event), int(v_q) / scale, int(u_q) / scale) for event, v_q, u_q in rows]
    assert len(trace) == n_steps, (
        f"Ibarz-Tanaka RTL emitted {len(trace)} trace rows; expected {n_steps}:\n"
        f"{simulation.stdout}"
    )
    return trace
