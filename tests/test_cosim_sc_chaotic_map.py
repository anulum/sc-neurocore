# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC chaotic-map Q8.24 co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.models.sc_chaotic_map_neuron import SCChaoticMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "src/sc_neurocore/neurons/model_schemas"
RTL = ROOT / "hdl/formal/catalogue/sc_chaotic_map.v"
SCALE = 1 << 24
Q_TEN = 10 * SCALE


def _lut() -> list[int]:
    source = RTL.read_text(encoding="utf-8")
    rows = re.findall(r"8'd(\d+): _sigmoid_lut1_out = 32'sd(\d+);", source)
    table = [0] * 256
    for index, value in rows:
        table[int(index)] = int(value)
    assert all(value > 0 for value in table)
    return table


def _fixed_trace(current: list[int]) -> list[tuple[int, int, int]]:
    table = _lut()
    x = y = 0
    rows = []
    for drive in current:
        previous_x = x
        raw_index = (x + 2 * SCALE + 16 * SCALE) >> 21
        sigmoid = table[max(0, min(255, raw_index))]
        fast_product = (11744051 * x) >> 24
        x = max(-Q_TEN, min(Q_TEN, ((fast_product * sigmoid) >> 24) - y + drive))
        y = max(-Q_TEN, min(Q_TEN, ((15938355 * y) >> 24) + ((838861 * previous_x) >> 24)))
        rows.append((int(previous_x < SCALE // 2 <= x), x, y))
    return rows


def _literal(value: int) -> str:
    return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"


def _rtl_trace(current: list[int]) -> list[tuple[int, int, int]]:
    assignments = "\n".join(
        f"    I_t = {_literal(value)}; @(posedge clk); #1; "
        f'$display("SCCM_TRACE %0d %0d %0d", spike_out, x_out, y_out);'
        for value in current
    )
    testbench = f"""`timescale 1ns/1ps
module tb;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [31:0] I_t = 32'sd0;
wire spike_out;
wire signed [31:0] x_out;
wire signed [31:0] y_out;
always #5 clk = ~clk;
sc_chaotic_map uut(.clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out), .x_out(x_out), .y_out(y_out));
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
        (int(event), int(x), int(y))
        for event, x, y in re.findall(
            r"^SCCM_TRACE (\d+) (-?\d+) (-?\d+)$", run.stdout, re.MULTILINE
        )
    ]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog unavailable")
def test_q824_is_bit_exact_to_quantized_project_specification() -> None:
    current = [SCALE, -SCALE] * 16
    observed = _rtl_trace(current)
    expected = _fixed_trace(current)
    assert observed == expected
    hand = SCChaoticMapNeuron()
    hand_rows = [(hand.step(value / SCALE), hand.x, hand.y) for value in current]
    assert [event for event, _, _ in observed] == [event for event, _, _ in hand_rows]
    for (_, fixed_x, fixed_y), (_, source_x, source_y) in zip(observed, hand_rows, strict=True):
        assert fixed_x / SCALE == pytest.approx(source_x, abs=0.009)
        assert fixed_y / SCALE == pytest.approx(source_y, abs=0.0016)


def test_paired_schemas_preserve_hand_recurrence() -> None:
    schemas = (
        UniversalNeuron.from_schema(SCHEMA_DIR / "sc_chaotic_map.toml"),
        UniversalNeuron.from_schema(SCHEMA_DIR / "sc_chaotic_map.json"),
    )
    hand = SCChaoticMapNeuron()
    for current in (0.1, 0.2, -0.1, 0.4, -0.3):
        event = hand.step(current)
        for schema in schemas:
            assert int(bool(schema.step(I=current))) == event
            assert schema.state == {"x": hand.x, "y": hand.y}
