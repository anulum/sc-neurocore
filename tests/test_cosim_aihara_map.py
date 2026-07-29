# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara Q8.24 bounded co-simulation

"""Measured short-horizon shadowing for the chaotic Aihara RTL."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.aihara_map_neuron import AiharaMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_aihara_map.v"


def _rtl_trace(n_steps: int) -> list[tuple[int, float]]:
    testbench = f"""`timescale 1ns/1ps
module tb_aihara;
reg clk = 1'b0;
reg rst_n = 1'b0;
wire spike_out;
wire signed [31:0] y_out;
always #5 clk = ~clk;
sc_aihara_map uut(.clk(clk), .rst_n(rst_n), .I_t(32'sd0), .spike_out(spike_out), .y_out(y_out));
integer index;
initial begin
  #23; rst_n = 1'b1;
  for (index = 0; index < {n_steps}; index = index + 1) begin
    @(posedge clk); #1; $display("AIHARA_TRACE %0d %0d", spike_out, y_out);
  end
  $finish;
end
endmodule
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tb = root / "tb.v"
        output = root / "tb"
        tb.write_text(testbench)
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
    rows = re.findall(r"^AIHARA_TRACE (\d+) (-?\d+)$", run.stdout, re.MULTILINE)
    assert len(rows) == n_steps
    scale = float(1 << 24)
    return [(int(event), int(y) / scale) for event, y in rows]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog unavailable")
def test_q824_shadows_source_events_and_state_for_twelve_steps() -> None:
    """Chaotic RTL gets a bounded horizon, not a false long-trace identity claim."""
    hand = AiharaMapNeuron()
    toml = UniversalNeuron.from_schema("aihara_map")
    expected: list[tuple[int, float]] = []
    for _ in range(12):
        event = hand.step(0.0)
        assert int(bool(toml.step(I=0.0))) == event
        assert toml.state["y"] == hand.y
        expected.append((event, hand.y))
    observed = _rtl_trace(12)
    assert [row[0] for row in observed] == [row[0] for row in expected]
    for (_, expected_y), (_, observed_y) in zip(expected, observed, strict=True):
        assert observed_y == pytest.approx(expected_y, abs=0.01)


def test_committed_rtl_contains_signed_lut_boundary_guards() -> None:
    """Prevent Verilog signedness regression in negative logistic arguments."""
    source = RTL.read_text()
    assert "$signed(_sigmoid_lut2_raw) < $signed(33'sd0)" in source
    assert "$signed({{_sigmoid_lut2_arg[31]}, _sigmoid_lut2_arg})" in source
