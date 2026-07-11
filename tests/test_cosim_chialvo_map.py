# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo map Python-to-Verilog co-simulation

"""Bounded Q16.16 evidence for the exponential Chialvo map."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float, float]
_CURRENTS = (-0.05, 0.0, 0.01, 0.1, 1.0)
_EXPECTED_EVENTS = {-0.05: 0, 0.0: 2, 0.01: 3, 0.1: 0, 1.0: 1}


def _signed_literal(value: int) -> str:
    if value < 0:
        return f"-32'sd{abs(value)}"
    return f"32'sd{value}"


def _rtl_q1616_trace(n_steps: int, current: float) -> list[TraceRow]:
    """Return the generated RTL event and two-state trace."""
    assert HAS_IVERILOG, "iverilog is required for Chialvo co-simulation"
    neuron = UniversalNeuron.from_schema("chialvo_map")
    module_name = "sc_chialvo_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_chialvo_map_q1616_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] x_out;",
            "wire signed [31:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t({_signed_literal(current_q)}),",
            "    .spike_out(spike_out), .x_out(x_out), .y_out(y_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            (
                f"    for (step_index = 0; step_index < {n_steps}; "
                "step_index = step_index + 1) begin"
            ),
            "        @(posedge clk); #1;",
            '        $display("CHIALVO_TRACE %0d %0d %0d", spike_out, x_out, y_out);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rtl_path = root / f"{module_name}.v"
        testbench_path = root / f"tb_{module_name}.v"
        executable = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        testbench_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(executable), str(rtl_path), str(testbench_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(executable)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

    rows = re.findall(
        r"^CHIALVO_TRACE (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    scale = float(1 << 16)
    trace = [(int(event), int(x_q) / scale, int(y_q) / scale) for event, x_q, y_q in rows]
    assert len(trace) == n_steps, (
        f"Chialvo RTL emitted {len(trace)} rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _hand_schema_trace(current: float, n_steps: int) -> list[TraceRow]:
    """Return exact hand/TOML/JSON evidence for the simultaneous recurrence."""
    hand = ChialvoMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIRECTORY / "chialvo_map.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIRECTORY / "chialvo_map.json")
    trace: list[TraceRow] = []
    for _step in range(n_steps):
        hand_event = hand.step(current)
        toml_event = int(bool(toml_schema.step(I=current)))
        json_event = int(bool(json_schema.step(I=current)))
        assert toml_event == json_event == hand_event
        assert toml_schema.state == json_schema.state == {"x": hand.x, "y": hand.y}
        trace.append((hand_event, hand.x, hand.y))
    return trace


def test_hand_toml_and_json_are_exact_at_enrolled_operating_set() -> None:
    """Both authored schemas must equal the hand source recurrence exactly."""
    for current in _CURRENTS:
        trace = _hand_schema_trace(current, 100)
        assert sum(row[0] for row in trace) == _EXPECTED_EVENTS[current]


def test_q1616_event_class_and_stable_trajectory_envelope() -> None:
    """Q16.16 must retain event counts and bounded stable-point states."""
    stable_currents = {-0.05, 0.1, 1.0}
    for current in _CURRENTS:
        hand_trace = _hand_schema_trace(current, 100)
        rtl_trace = _rtl_q1616_trace(100, current)
        assert sum(row[0] for row in rtl_trace) == _EXPECTED_EVENTS[current]
        if current not in stable_currents:
            continue
        x_error = max(
            abs(expected[1] - observed[1])
            for expected, observed in zip(hand_trace, rtl_trace, strict=True)
        )
        y_error = max(
            abs(expected[2] - observed[2])
            for expected, observed in zip(hand_trace, rtl_trace, strict=True)
        )
        assert x_error < 0.055
        assert y_error < 0.093


def test_q1616_oscillatory_event_timing_is_declared_boundary() -> None:
    """LUT quantisation may phase-shift events without changing their count."""
    expected_mismatches = {0.0: 4, 0.01: 6}
    for current, expected_count in expected_mismatches.items():
        hand_trace = _hand_schema_trace(current, 100)
        rtl_trace = _rtl_q1616_trace(100, current)
        mismatches = sum(
            expected[0] != observed[0]
            for expected, observed in zip(hand_trace, rtl_trace, strict=True)
        )
        assert mismatches == expected_count
