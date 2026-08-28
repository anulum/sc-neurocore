# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cazelles source-map Python-to-Verilog co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

TraceRow = tuple[int, float]


def _rtl_q1616_trace(n_steps: int) -> list[TraceRow]:
    assert HAS_IVERILOG
    neuron = UniversalNeuron.from_schema("cazelles_map")
    module_name = "sc_cazelles_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(0.0)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_cazelles_map_q1616_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] x_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .x_out(x_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("CAZELLES_TRACE %0d %0d", spike_out, uut.x_reg);',
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
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        simulation = subprocess.run(
            ["vvp", str(out_path)], check=True, capture_output=True, text=True, timeout=30
        )
    rows = re.findall(r"^CAZELLES_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << 16)
    result = [(int(event), int(x_q) / scale) for event, x_q in rows]
    assert len(result) == n_steps
    return result


def _hand_schema_trace(n_steps: int) -> tuple[list[TraceRow], dict[int, int]]:
    hand = CazellesMapNeuron()
    toml = UniversalNeuron.from_schema("cazelles_map")
    json_model = UniversalNeuron.from_schema(
        Path(__file__).resolve().parents[1]
        / "src/sc_neurocore/neurons/model_schemas/cazelles_map.json"
    )
    trace: list[TraceRow] = []
    branches = {1: 0, 2: 0, 3: 0, 4: 0}
    for _ in range(n_steps):
        branch = 1 if hand.x < hand.x1 else 2 if hand.x < hand.x2 else 3 if hand.x < hand.x3 else 4
        branches[branch] += 1
        event = hand.step(0.0)
        assert toml.step(I=0.0) == event
        assert json_model.step(I=0.0) == event
        assert toml.state == json_model.state == {"x": hand.x}
        trace.append((event, hand.x))
    return trace, branches


def test_hand_toml_json_cover_all_source_branches() -> None:
    trace, branches = _hand_schema_trace(600)
    assert branches == {1: 349, 2: 122, 3: 7, 4: 122}
    assert sum(event for event, _x in trace) == 7


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog is required")
def test_q1616_tracks_source_orbit_until_first_discontinuous_return() -> None:
    hand, _branches = _hand_schema_trace(55)
    rtl = _rtl_q1616_trace(55)
    assert [event for event, _x in rtl] == [event for event, _x in hand]
    errors = [
        abs(observed - expected)
        for (_event, expected), (_event2, observed) in zip(hand, rtl, strict=True)
    ]
    assert max(errors) <= 0.0062


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog is required")
def test_q1616_long_chaotic_orbit_is_an_explicit_excluded_boundary() -> None:
    hand, _branches = _hand_schema_trace(600)
    rtl = _rtl_q1616_trace(600)
    assert all(0.0 <= x <= 1.0 for _event, x in rtl)
    assert sum(event for event, _x in hand) == 7
    assert sum(event for event, _x in rtl) == 2
