# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — four-site Amari field Q16.16 co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from tests.cosim_support import HAS_IVERILOG

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_amari_field.v"
SCALE = 1 << 16
WEIGHTS = (49152, 6352, -4778)
DX = 32768
STEP = 3277
LIMIT = 8 * SCALE


def _drives(steps: int) -> list[list[int]]:
    values = []
    for step in range(steps):
        row = [-0.18, -0.18, -0.18, -0.18]
        row[step % 4] = 0.42
        if step % 7 == 0:
            row[(step + 2) % 4] = 0.11
        values.append([round(value * SCALE) for value in row])
    return values


def _fixed_trace(drives: list[list[int]]) -> list[tuple[int, int, int, int, int]]:
    state = [0, 0, 0, 0]
    rows = []
    for drive in drives:
        active = [value > 0 for value in state]
        candidate = []
        for site in range(4):
            interaction = 0
            for source in range(4):
                if active[source]:
                    distance = min((site - source) % 4, (source - site) % 4)
                    interaction += WEIGHTS[distance]
            convolution = (interaction * DX) >> 16
            delta = ((-state[site] + convolution + drive[site]) * STEP) >> 16
            candidate.append(max(-LIMIT, min(LIMIT, state[site] + delta)))
        state = candidate
        rate = sum(value > 0 for value in state) * (SCALE // 4)
        rows.append((state[0], state[1], state[2], state[3], rate))
    return rows


def _literal(value: int) -> str:
    return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"


def _rtl_trace(drives: list[list[int]]) -> list[tuple[int, int, int, int, int]]:
    assignments = "\n".join(
        "    "
        + "; ".join(f"I{site}_t = {_literal(value)}" for site, value in enumerate(row))
        + '; @(posedge clk); #1; $display("AMARI_TRACE %0d %0d %0d %0d %0d", '
        + "u0, u1, u2, u3, rate);"
        for row in drives
    )
    testbench = f"""`timescale 1ns/1ps
module tb;
reg clk=0; reg rst_n=0;
reg signed [31:0] I0_t=0, I1_t=0, I2_t=0, I3_t=0;
wire signed [31:0] u0,u1,u2,u3; wire [31:0] rate;
always #5 clk=~clk;
sc_amari_field uut(.clk(clk),.rst_n(rst_n),.I0_t(I0_t),.I1_t(I1_t),.I2_t(I2_t),.I3_t(I3_t),
 .u0_out(u0),.u1_out(u1),.u2_out(u2),.u3_out(u3),.mean_rate_out(rate));
initial begin
  #23; rst_n=1;
{assignments}
  $finish;
end
endmodule
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tb = root / "tb.v"
        binary = root / "tb"
        tb.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(binary), str(RTL), str(tb)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        completed = subprocess.run(
            ["vvp", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    return [
        (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))
        for row in re.findall(
            r"^AMARI_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\d+)$",
            completed.stdout,
            re.MULTILINE,
        )
    ]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog unavailable")
def test_q1616_rtl_is_bit_exact_and_tracks_source_field() -> None:
    drives = _drives(64)
    observed = _rtl_trace(drives)
    expected = _fixed_trace(drives)
    assert observed == expected
    hand = AmariNeuralField(n=4)
    float_rows: list[tuple[float, float, float, float, float]] = []
    for row in drives:
        rate = hand.step(np.asarray(row, dtype=np.float64) / SCALE)
        state = hand.u
        assert state is not None
        float_rows.append((state[0], state[1], state[2], state[3], rate))
    assert [row[-1] for row in observed] == [round(row[-1] * SCALE) for row in float_rows]
    for fixed, source in zip(observed, float_rows, strict=True):
        np.testing.assert_allclose(np.asarray(fixed[:4]) / SCALE, source[:4], rtol=0.0, atol=0.0025)


def test_rtl_declares_numeric_interface_and_rate_semantics() -> None:
    source = RTL.read_text(encoding="utf-8")
    assert "signed Q16.16" in source
    assert "population rate, not a spike" in source
    assert "Latency is one cycle" in source
