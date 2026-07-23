# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI neuron co-simulation references

"""Independent DPI neuron spike-count and driven-Euler reference contracts."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_reference_statistics import _summarise


def _dpi_neuron_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored DPI (current-mode Euler) spike count for comparison."""
    neuron = DPINeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _dpi_neuron_driven_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return independent features for the coupled published DPI equations.

    Indiveri, Stefanini, and Chicca (2010), Eqs. (2)–(3), define the nonlinear
    positive-feedback membrane current and the spike-triggered adaptation DPI.
    This helper re-derives both right-hand sides directly, advances them
    simultaneously with explicit Euler, holds ``i_mem`` at reset during the
    refractory pulse, and applies the post-update threshold/reset ordering. It
    intentionally does not call the maintained model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for all three states plus event features.
    """
    if steps <= 0:
        raise ValueError("DPI reference trace requires at least one step")
    i_threshold = 1.0
    i_reset = 0.01
    i_rest = 0.1
    i_tau = 1.0
    i_g = 1.0
    i_tau_ahp = 0.1
    i_ga = 1.0
    i_spike = 5.0
    i_0 = 0.01
    kappa = 0.7
    alpha = 10.0
    tau = 20.0
    tau_ahp = 100.0
    refractory_period = 2.0
    i_mem = 0.01
    i_ahp = 0.01
    refractory_time = 0.0
    i_mem_values: list[float] = []
    i_ahp_values: list[float] = []
    refractory_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        spike_active = refractory_time > 0.0
        spike_current = i_spike if spike_active else 0.0
        d_i_ahp = i_ahp / (tau_ahp * i_tau_ahp) * (spike_current / (1.0 + i_ahp / i_ga) - i_tau_ahp)
        next_i_ahp = i_ahp + dt * d_i_ahp
        if spike_active:
            event = 0
            next_i_mem = i_reset
            next_refractory = max(0.0, refractory_time - dt)
        else:
            log_current = (math.log(i_0) + kappa * math.log(i_mem)) / (kappa + 1.0)
            # Every candidate at or above threshold resets before the next iteration.
            gate_argument = alpha * (i_mem - i_threshold)
            exponential = math.exp(gate_argument)
            gate = exponential / (1.0 + exponential)
            i_fb = math.exp(log_current) * gate
            d_i_mem = (
                i_mem
                / (tau * i_tau)
                * ((i_rest + current) / (1.0 + i_mem / i_g) - i_tau + i_fb - i_ahp)
            )
            next_i_mem = i_mem + dt * d_i_mem
            event = int(next_i_mem >= i_threshold)
            next_refractory = 0.0
            if event:
                next_i_mem = i_reset
                next_refractory = refractory_period
        i_mem, i_ahp, refractory_time = next_i_mem, next_i_ahp, next_refractory
        spikes.append(event)
        i_mem_values.append(i_mem)
        i_ahp_values.append(i_ahp)
        refractory_values.append(refractory_time)

    return _summarise(
        {
            "i_mem": i_mem_values,
            "i_ahp": i_ahp_values,
            "refractory_time": refractory_values,
        },
        spikes,
    )


def _dpi_neuron_verilog_q1616_trace(
    n_steps: int,
    current: float,
) -> list[tuple[int, float, float, float]]:
    """Return the emitted DPI RTL's committed Q16.16 three-state trace.

    Reset is deasserted between clock edges, so the first sampled rising edge is
    exactly logical step zero. This avoids the generic generated testbench's
    deliberate uncounted settling edge and makes event timing and state rows
    directly comparable to consecutive ``DPINeuron.step`` calls.
    """
    neuron = UniversalNeuron.from_schema("dpi_neuron")
    module_name = "sc_dpi_neuron_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] i_mem_out;",
            "wire signed [31:0] i_ahp_out;",
            "wire signed [31:0] refractory_time_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .i_mem_out(i_mem_out),",
            "    .i_ahp_out(i_ahp_out), .refractory_time_out(refractory_time_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("DPI_Q1616_TRACE %0d %0d %0d %0d",',
            "            spike_out, i_mem_out, i_ahp_out, refractory_time_out);",
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
            timeout=60,
            check=True,
        )

    scale = float(1 << 16)
    rows = re.findall(
        r"^DPI_Q1616_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    trace = [
        (
            int(event),
            int(i_mem_q) / scale,
            int(i_ahp_q) / scale,
            int(refractory_q) / scale,
        )
        for event, i_mem_q, i_ahp_q, refractory_q in rows
    ]
    assert len(trace) == n_steps, (
        f"DPI RTL emitted {len(trace)} trace rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace
