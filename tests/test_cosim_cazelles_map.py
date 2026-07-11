# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cazelles map Python-to-Verilog co-simulation

"""Bounded Q16.16 trajectory evidence for the Cazelles fast/slow map."""

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

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float, float]


def _rtl_q1616_trace(n_steps: int, current: float) -> list[TraceRow]:
    """Return the generated RTL's committed event and two-state trace."""
    assert HAS_IVERILOG, "iverilog is required for Cazelles co-simulation"
    neuron = UniversalNeuron.from_schema("cazelles_map")
    module_name = "sc_cazelles_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_cazelles_map_q1616_trace;",
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
            '        $display("CAZELLES_TRACE %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);',
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

    rows = re.findall(
        r"^CAZELLES_TRACE (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    scale = float(1 << 16)
    trace = [(int(event), int(x_q) / scale, int(y_q) / scale) for event, x_q, y_q in rows]
    assert len(trace) == n_steps, (
        f"Cazelles RTL emitted {len(trace)} rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _hand_schema_trace(current: float, n_steps: int) -> tuple[list[TraceRow], dict[str, int]]:
    """Return exact hand/schema trace evidence and fast-map branch counts."""
    hand = CazellesMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "cazelles_map.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "cazelles_map.json")
    trace: list[TraceRow] = []
    clamp_counts = {"low": 0, "interior": 0, "high": 0}

    for _step in range(n_steps):
        raw_x = hand.a * hand.x * (1.0 - hand.x) - hand.y + current
        branch = "low" if raw_x < -2.0 else "high" if raw_x > 2.0 else "interior"
        clamp_counts[branch] += 1
        hand_event = hand.step(current)
        assert int(bool(toml_schema.step(I=current))) == hand_event
        assert int(bool(json_schema.step(I=current))) == hand_event
        assert toml_schema.state == {"x": hand.x, "y": hand.y}
        assert json_schema.state == {"x": hand.x, "y": hand.y}
        trace.append((hand_event, hand.x, hand.y))

    return trace, clamp_counts


@pytest.mark.parametrize(
    ("current", "expected_events", "expected_clamps"),
    (
        (0.5, 2, {"low": 25, "interior": 5, "high": 0}),
        (1.0, 1, {"low": 28, "interior": 2, "high": 0}),
        (2.0, 1, {"low": 29, "interior": 0, "high": 1}),
    ),
    ids=("interior-and-low-clip", "low-clip", "both-clip-bounds"),
)
def test_q1616_short_window_trajectory(
    current: float,
    expected_events: int,
    expected_clamps: dict[str, int],
) -> None:
    """RTL must preserve the bounded event vector and track both coordinates."""
    n_steps = 30
    hand_trace, clamp_counts = _hand_schema_trace(current, n_steps)
    rtl_trace = _rtl_q1616_trace(n_steps, current)

    assert clamp_counts == expected_clamps
    assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
    assert sum(row[0] for row in rtl_trace) == expected_events
    for (_event, expected_x, expected_y), (_rtl_event, rtl_x, rtl_y) in zip(
        hand_trace, rtl_trace, strict=True
    ):
        assert rtl_x == pytest.approx(expected_x, abs=0.0004)
        assert rtl_y == pytest.approx(expected_y, abs=0.0004)


def test_q1616_declares_sensitive_boundary() -> None:
    """The ``I=0.05`` chaotic trajectory must remain outside the parity band."""
    hand_trace, _clamp_counts = _hand_schema_trace(current=0.05, n_steps=30)
    rtl_trace = _rtl_q1616_trace(n_steps=30, current=0.05)
    event_mismatches = sum(
        expected[0] != observed[0] for expected, observed in zip(hand_trace, rtl_trace, strict=True)
    )

    assert sum(row[0] for row in hand_trace) == 7
    assert sum(row[0] for row in rtl_trace) == 8
    assert event_mismatches == 7
