# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Nagumo–Sato Q16.16 co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.nagumo_sato_map_neuron import NagumoSatoMapNeuron
from tests.cosim_support import HAS_IVERILOG

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_nagumo_sato_map.v"
SCALE = 1 << 16


def _saturate(value: int) -> int:
    return max(-(1 << 31), min((1 << 31) - 1, value))


def _fixed_trace(current: list[int]) -> list[tuple[int, int]]:
    y = 6554
    rows = []
    for drive in current:
        y = _saturate((39322 * y >> 16) - (65536 if y >= 0 else 0) + 13107 + drive)
        rows.append((int(y >= 0), y))
    return rows


def _literal(value: int) -> str:
    return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"


def _rtl_trace(current: list[int]) -> list[tuple[int, int]]:
    assignments = "\n".join(
        f"    I_t = {_literal(value)}; @(posedge clk); #1; "
        f'$display("NS_TRACE %0d %0d", spike_out, y_out);'
        for value in current
    )
    testbench = f"""`timescale 1ns/1ps
module tb;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [31:0] I_t = 32'sd0;
wire spike_out;
wire signed [31:0] y_out;
always #5 clk = ~clk;
sc_nagumo_sato_map uut(.clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out), .y_out(y_out));
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
        (int(event), int(y))
        for event, y in re.findall(r"^NS_TRACE (\d+) (-?\d+)$", run.stdout, re.MULTILINE)
    ]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog unavailable")
def test_q1616_is_bit_exact_to_quantized_source_equation() -> None:
    current = [round((0.05 * ((index % 7) - 3)) * SCALE) for index in range(64)]
    observed = _rtl_trace(current)
    expected = _fixed_trace(current)
    assert observed == expected
    hand = NagumoSatoMapNeuron()
    hand_rows = [(hand.step(value / SCALE), hand.y) for value in current]
    assert [event for event, _ in observed] == [event for event, _ in hand_rows]
    for (_, fixed_y), (_, source_y) in zip(observed, hand_rows, strict=True):
        assert fixed_y / SCALE == pytest.approx(source_y, abs=5e-5)
