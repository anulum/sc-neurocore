# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC adaptive-threshold-map Q8.24 co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.sc_adaptive_threshold_map_neuron import (
    SCAdaptiveThresholdMapNeuron,
)
from tests.cosim_support import HAS_IVERILOG

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_adaptive_threshold_map.v"
SCALE = 1 << 24
Q_FIVE = 5 * SCALE


def _lut() -> list[int]:
    source = RTL.read_text(encoding="utf-8")
    rows = re.findall(r"8'd(\d+): _sigmoid_lut2_out = 32'sd(\d+);", source)
    table = [0] * 256
    for index, value in rows:
        table[int(index)] = int(value)
    assert all(value > 0 for value in table)
    return table


def _fixed_trace(current: list[int]) -> list[tuple[int, int, int]]:
    table = _lut()
    x = theta = 0
    rows = []
    for drive in current:
        previous_x = x
        raw_index = (((x - theta) << 2) + 16 * SCALE) >> 21
        activation = table[max(0, min(255, raw_index))]
        x = max(-Q_FIVE, min(Q_FIVE, -x + (25165824 * activation >> 24) + drive))
        theta = max(
            -Q_FIVE,
            min(Q_FIVE, (15938355 * theta >> 24) + (5033165 if previous_x >= 13421773 else 0)),
        )
        event = int(x >= 13421773 and previous_x < 13421773)
        rows.append((event, x, theta))
    return rows


def _literal(value: int) -> str:
    return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"


def _rtl_trace(current: list[int]) -> list[tuple[int, int, int]]:
    assignments = "\n".join(
        f"    I_t = {_literal(value)}; @(posedge clk); #1; "
        f'$display("SCAT_TRACE %0d %0d %0d", spike_out, x_out, theta_out);'
        for value in current
    )
    testbench = f"""`timescale 1ns/1ps
module tb;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [31:0] I_t = 32'sd0;
wire spike_out;
wire signed [31:0] x_out;
wire signed [31:0] theta_out;
always #5 clk = ~clk;
sc_adaptive_threshold_map uut(.clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out), .x_out(x_out), .theta_out(theta_out));
initial begin
  #23; rst_n = 1'b1;
{assignments}
  $finish;
end
endmodule
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tb = root / "tb.v"
        output = root / "tb"
        tb.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(output), str(RTL), str(tb)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        run = subprocess.run(
            ["vvp", str(output)], check=True, capture_output=True, text=True, timeout=30
        )
    return [
        (int(event), int(x), int(theta))
        for event, x, theta in re.findall(
            r"^SCAT_TRACE (\d+) (-?\d+) (-?\d+)$", run.stdout, re.MULTILINE
        )
    ]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog unavailable")
def test_q824_is_bit_exact_to_quantized_project_specification() -> None:
    current = [round(0.4 * SCALE)] * 32
    observed = _rtl_trace(current)
    expected = _fixed_trace(current)
    assert observed == expected
    hand = SCAdaptiveThresholdMapNeuron()
    hand_rows = [(hand.step(value / SCALE), hand.x, hand.theta) for value in current]
    assert [event for event, _, _ in observed] == [event for event, _, _ in hand_rows]
    for (_, fixed_x, fixed_theta), (_, source_x, source_theta) in zip(
        observed, hand_rows, strict=True
    ):
        assert fixed_x / SCALE == pytest.approx(source_x, abs=0.07)
        assert fixed_theta / SCALE == pytest.approx(source_theta, abs=2e-6)
