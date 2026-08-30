# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF Python-to-Verilog fidelity contracts

"""Exact-flow hand model, paired schema, and Q16.16 RTL contracts for QIF."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.quadratic_if import (
    QuadraticIFNeuron,
    SCSymmetricQuadraticIFNeuron,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_Q1616_CASES = (
    (0.0, 0),
    (0.333, 2),
    (0.5, 3),
    (1.0, 6),
    (2.0, 11),
    (5.0, 26),
    (20.0, 100),
    (50.0, 250),
)


def _q1616_rtl_trace(current: float, n_steps: int) -> list[tuple[int, float]]:
    """Return the generated RTL's committed event and voltage trace."""
    neuron = UniversalNeuron.from_schema("sc_symmetric_quadratic_if")
    module_name = "sc_quadratic_if_euler_q1616"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] v_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .v_out(v_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            (
                f"    for (step_index = 0; step_index < {n_steps}; "
                "step_index = step_index + 1) begin"
            ),
            "        @(posedge clk); #1;",
            '        $display("QIF_TRACE %0d %0d", spike_out, uut.v_reg);',
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
    rows = re.findall(r"^QIF_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << 16)
    trace = [(int(event), int(voltage_q) / scale) for event, voltage_q in rows]
    assert len(trace) == n_steps
    return trace


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_preserve_exact_flow_events_and_state_envelope() -> None:
    """The retained SC schemas track their exact-flow hand compatibility model."""
    hand = SCSymmetricQuadraticIFNeuron()
    toml_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "sc_symmetric_quadratic_if.toml")
    json_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "sc_symmetric_quadratic_if.json")
    currents = (0.0, 0.333, 0.5, 1.0, 2.0, 5.0, 20.0, 50.0) * 125
    events = 0
    max_error = 0.0
    for current in currents:
        hand_event = hand.step(current)
        events += hand_event
        assert int(bool(toml_schema.step(I=current))) == hand_event
        assert int(bool(json_schema.step(I=current))) == hand_event
        assert toml_schema.state["v"] == json_schema.state["v"]
        max_error = max(max_error, abs(toml_schema.state["v"] - hand.v))
    assert events == 41
    assert max_error < 0.006


@pytest.mark.parametrize("suffix", ("toml", "json"))
def test_schema_threshold_equality_uses_configured_peak(suffix: str) -> None:
    """Reset when the Euler candidate equals the configured peak exactly."""
    schema = UniversalNeuron.from_schema(
        _SCHEMA_DIR / f"sc_symmetric_quadratic_if.{suffix}",
        parameter_overrides={"v_peak": 2.0},
    )
    assert bool(schema.step(I=299.0))
    assert schema.state["v"] == -1.0


@pytest.mark.parametrize(
    ("current", "expected_events"),
    _Q1616_CASES,
    ids=[f"I={current:g}" for current, _events in _Q1616_CASES],
)
def test_q1616_preserves_event_vectors_and_voltage_bound(
    current: float,
    expected_events: int,
) -> None:
    """Preserve exact event timing and a measured Q16.16 voltage envelope."""
    n_steps = 1_000
    hand = SCSymmetricQuadraticIFNeuron()
    schema = UniversalNeuron.from_schema("sc_symmetric_quadratic_if")
    hand_trace = [(hand.step(current), hand.v) for _ in range(n_steps)]
    schema_trace = [(int(bool(schema.step(I=current))), schema.state["v"]) for _ in range(n_steps)]
    rtl_trace = _q1616_rtl_trace(current, n_steps)
    hand_events = [event for event, _voltage in hand_trace]
    assert [event for event, _voltage in schema_trace] == hand_events
    assert [event for event, _voltage in rtl_trace] == hand_events
    assert sum(hand_events) == expected_events
    assert (
        max(
            abs(rtl_voltage - hand_voltage)
            for (_event, hand_voltage), (_rtl_event, rtl_voltage) in zip(
                hand_trace, rtl_trace, strict=True
            )
        )
        < 0.011
    )


def test_q1616_declares_low_drive_reset_timing_boundary() -> None:
    """Pin the I=0.1 one-cycle RTL reset displacement instead of hiding it."""
    current = 0.1
    n_steps = 1_000
    hand = SCSymmetricQuadraticIFNeuron()
    schema = UniversalNeuron.from_schema("sc_symmetric_quadratic_if")
    hand_events = [hand.step(current) for _ in range(n_steps)]
    schema_events = [int(bool(schema.step(I=current))) for _ in range(n_steps)]
    rtl_events = [event for event, _voltage in _q1616_rtl_trace(current, n_steps)]
    assert hand_events == schema_events
    assert (sum(hand_events), sum(rtl_events)) == (1, 1)
    assert sum(hand != rtl for hand, rtl in zip(hand_events, rtl_events, strict=True)) == 2


def _source_q1616_rtl_trace(current: float, n_steps: int) -> list[tuple[int, float]]:
    """Execute the tracked Latham-profile Q16.16 RTL representative."""
    repository = Path(__file__).resolve().parents[1]
    rtl_path = repository / "hdl/formal/catalogue/sc_quadratic_if_latham_2000.v"
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_quadratic_if_latham_2000;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] v_out;",
            "always #5 clk = ~clk;",
            "sc_quadratic_if_latham_2000 uut (",
            "  .clk(clk), .rst_n(rst_n),",
            f"  .I_t(32'sd{current_q}), .spike_out(spike_out), .v_out(v_out));",
            "integer step_index;",
            "initial begin",
            "  #23; rst_n = 1'b1;",
            f"  for (step_index=0; step_index<{n_steps}; step_index=step_index+1) begin",
            "    @(posedge clk); #1;",
            '    $display("QIF_SOURCE %0d %0d", spike_out, uut.v_reg);',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tb_path = root / "tb.v"
        out_path = root / "tb"
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
    rows = re.findall(r"^QIF_SOURCE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    assert len(rows) == n_steps
    scale = float(1 << 16)
    return [(int(event), int(voltage) / scale) for event, voltage in rows]


@pytest.mark.parametrize("current", (0.0, 2.0, 4.0, 8.0))
def test_latham_source_rtl_tracks_source_euler_schema(current: float) -> None:
    """Keep source boundaries and Euler timestep cycle-aligned in Q16.16."""
    n_steps = 240
    schema = UniversalNeuron.from_schema("quadratic_if")
    schema_trace = [(int(bool(schema.step(I=current))), schema.state["v"]) for _ in range(n_steps)]
    rtl_trace = _source_q1616_rtl_trace(current, n_steps)
    assert [event for event, _ in rtl_trace] == [event for event, _ in schema_trace]
    max_error = max(
        abs(rtl_voltage - schema_voltage)
        for (_rtl_event, rtl_voltage), (_schema_event, schema_voltage) in zip(
            rtl_trace, schema_trace, strict=True
        )
    )
    assert max_error < 0.003


def test_latham_source_rtl_declares_q1616_one_cycle_boundary() -> None:
    """Pin the I=1 quantization timing displacement without hiding it."""
    schema = UniversalNeuron.from_schema("quadratic_if")
    schema_events = [int(bool(schema.step(I=1.0))) for _ in range(240)]
    rtl_events = [event for event, _voltage in _source_q1616_rtl_trace(1.0, 240)]
    assert sum(schema_events) == sum(rtl_events) == 4
    assert sum(a != b for a, b in zip(schema_events, rtl_events, strict=True)) == 8


def test_latham_exact_runtime_and_source_euler_are_not_conflated() -> None:
    """Declare the maintained exact-flow specialization separately from source Euler RTL."""
    exact = QuadraticIFNeuron.latham_2000()
    schema = UniversalNeuron.from_schema("quadratic_if")
    exact_events = [exact.step(4.0) for _ in range(240)]
    schema_events = [int(bool(schema.step(I=4.0))) for _ in range(240)]
    assert sum(exact_events) == 10
    assert sum(schema_events) == 9
    assert exact_events != schema_events
