# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map co-simulation references

"""Independent Rulkov map RTL-trace and piecewise-map reference contracts."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_reference_statistics import _summarise


def _rulkov_map_verilog_q1616_trace(n_steps: int, current: float) -> list[tuple[int, float, float]]:
    """Return the emitted Rulkov RTL's committed Q16.16 state trace.

    The testbench samples the generated module's synchronous ``x_reg`` and
    ``y_reg`` state after each active clock edge. These registers are the map
    recurrence itself; the public state outputs retain the pre-threshold value
    on a spiking cycle, so sampling the committed registers
    avoids confusing that interface convention with the next-state trajectory.
    """
    neuron = UniversalNeuron.from_schema("rulkov_map")
    module_name = "sc_rulkov_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_rulkov_map_q1616_trace;",
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
            '        $display("RULKOV_TRACE %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);',
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
    rows = re.findall(r"^RULKOV_TRACE (-?\d+) (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    trace = [(int(spike), int(x_q) / scale, int(y_q) / scale) for spike, x_q, y_q in rows]
    assert len(trace) == n_steps, (
        f"Rulkov RTL emitted {len(trace)} trace rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _rulkov_map_features(*, current: float, steps: int) -> dict[str, float]:
    """Return exact features for the Rulkov 2002 piecewise map iteration.

    The Rulkov (2002) fast/slow model is a discrete map, so an independent
    implementation of its three-branch fast map (rational subthreshold, spike
    plateau, hard reset) and slow drift reproduces the runner exactly — a map has no
    integration error, so independent parity is exact ground truth. Upward-crossing
    detection (post-update ``x >= 0`` with pre-update ``x < 0``) matches the hand
    model and schema runner.

    Parameters
    ----------
    current:
        Constant drive applied at every iteration.
    steps:
        Number of map iterations to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``x`` and ``y`` state variables plus
        spike-count and first-spike-step features.
    """
    alpha = 4.0
    sigma = -1.6
    mu = 0.001
    x = -1.0
    y = -3.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        x_previous = x
        if x <= 0:
            x_next = alpha / (1.0 - x) + y + current
        elif x < alpha + y + current:
            x_next = alpha + y + current
        else:
            x_next = -1.0
        y_next = y - mu * (x + 1.0) + mu * sigma
        x, y = x_next, y_next
        spikes.append(1 if x >= 0.0 and x_previous < 0.0 else 0)
        x_values.append(x)
        y_values.append(y)

    return _summarise({"x": x_values, "y": y_values}, spikes)
