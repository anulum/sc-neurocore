# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta Python-to-Verilog fidelity contracts

"""Exact-flow hand model, paired Euler schemas, and Q16.16 RTL contracts."""

from __future__ import annotations

import math
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.theta import ThetaNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_Q1616_CASES = (
    (-1.0, 0),
    (-0.5, 0),
    (0.0, 0),
    (0.1, 1),
    (0.333, 2),
    (0.5, 2),
    (1.0, 3),
    (2.0, 5),
    (5.0, 7),
    (20.0, 14),
    (50.0, 23),
)
_PHASE_ENVELOPE_CURRENTS = (-1.0, -0.5, 0.0, 0.333, 0.5, 1.0, 2.0)
_Q1616_PHASE_ATOL = 0.17
_REPOSITORY = Path(__file__).resolve().parents[1]


def _circular_error(actual: float, expected: float) -> float:
    """Return the shortest absolute distance between two circle phases."""
    return abs((actual - expected + math.pi) % (2.0 * math.pi) - math.pi)


def _q1616_rtl_trace(current: float, n_steps: int) -> list[tuple[int, float]]:
    """Return the generated Q16.16 RTL event and committed phase trace."""
    neuron = UniversalNeuron.from_schema("theta")
    module_name = "sc_theta_euler_q1616"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] theta_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .theta_out(theta_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            (
                f"    for (step_index = 0; step_index < {n_steps}; "
                "step_index = step_index + 1) begin"
            ),
            "        @(posedge clk); #1;",
            '        $display("THETA_TRACE %0d %0d", spike_out, uut.theta_reg);',
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
    rows = re.findall(r"^THETA_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << 16)
    trace = [(int(event), int(theta_q) / scale) for event, theta_q in rows]
    assert len(trace) == n_steps
    return trace


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


@pytest.mark.parametrize(
    ("current", "expected_events"),
    _Q1616_CASES,
    ids=[f"I={current:g}" for current, _events in _Q1616_CASES],
)
def test_schema_formats_preserve_exact_flow_event_counts(
    current: float,
    expected_events: int,
) -> None:
    """Keep TOML and JSON identical and count-equivalent to the exact hand flow."""
    n_steps = 1_000
    hand = ThetaNeuron()
    toml_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "theta.toml")
    json_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "theta.json")
    hand_events: list[int] = []
    toml_events: list[int] = []
    json_events: list[int] = []
    for _step in range(n_steps):
        hand_events.append(hand.step(current))
        toml_events.append(int(bool(toml_schema.step(I=current))))
        json_events.append(int(bool(json_schema.step(I=current))))
        assert toml_schema.state["theta"] == json_schema.state["theta"]
    assert sum(hand_events) == expected_events
    assert sum(toml_events) == expected_events
    assert json_events == toml_events


@pytest.mark.parametrize(
    ("current", "expected_events"),
    _Q1616_CASES,
    ids=[f"I={current:g}" for current, _events in _Q1616_CASES],
)
def test_q1616_preserves_complete_event_count_vector(
    current: float,
    expected_events: int,
) -> None:
    """Preserve the enrolled exact-flow event counts through generated RTL."""
    n_steps = 1_000
    hand = ThetaNeuron()
    schema = UniversalNeuron.from_schema("theta")
    hand_events = [hand.step(current) for _step in range(n_steps)]
    schema_events = [int(bool(schema.step(I=current))) for _step in range(n_steps)]
    rtl_events = [event for event, _phase in _q1616_rtl_trace(current, n_steps)]
    assert (sum(hand_events), sum(schema_events), sum(rtl_events)) == (
        expected_events,
        expected_events,
        expected_events,
    )


@pytest.mark.parametrize(
    "current",
    _PHASE_ENVELOPE_CURRENTS,
    ids=[f"I={current:g}" for current in _PHASE_ENVELOPE_CURRENTS],
)
def test_q1616_preserves_declared_circular_phase_envelope(current: float) -> None:
    """Bound generated phase error where event timing stays numerically stable."""
    n_steps = 1_000
    hand = ThetaNeuron()
    hand_trace = [(hand.step(current), hand.theta) for _step in range(n_steps)]
    rtl_trace = _q1616_rtl_trace(current, n_steps)
    assert (
        max(
            _circular_error(rtl_phase, hand_phase)
            for (_hand_event, hand_phase), (_rtl_event, rtl_phase) in zip(
                hand_trace, rtl_trace, strict=True
            )
        )
        < _Q1616_PHASE_ATOL
    )


def test_q1616_declares_one_cycle_event_timing_boundary() -> None:
    """Pin the bounded Euler/quantised timing shift instead of hiding it."""
    current = 1.0
    n_steps = 1_000
    hand = ThetaNeuron()
    schema = UniversalNeuron.from_schema("theta")
    hand_events = [hand.step(current) for _step in range(n_steps)]
    schema_events = [int(bool(schema.step(I=current))) for _step in range(n_steps)]
    rtl_events = [event for event, _phase in _q1616_rtl_trace(current, n_steps)]
    assert (sum(hand_events), sum(schema_events), sum(rtl_events)) == (3, 3, 3)
    assert sum(hand != rtl for hand, rtl in zip(hand_events, rtl_events, strict=True)) == 6


def test_committed_yosys_report_proves_nontrivial_q1616_synthesis() -> None:
    """Bind H2 to the executed coarse-synthesis receipt for the committed RTL."""
    report = json.loads(
        (_REPOSITORY / "hdl/reports/yosys_theta_q1616_2026-08-30.json").read_text(encoding="utf-8")
    )
    module = report["modules"]["\\sc_theta"]
    assert module["num_processes"] == 0
    assert module["num_cells"] == 6203
    assert module["num_cells_by_type"]["$_DFF_PN0_"] == 33
    assert module["num_cells_by_type"]["$_MUX_"] == 381


def test_curated_formal_job_reaches_and_checks_receipt_drive_event() -> None:
    """Pin the model-specific proof depth, drive, phase envelope, and wrap."""
    formal_dir = _REPOSITORY / "hdl/formal/catalogue"
    job = (formal_dir / "sc_theta.sby").read_text(encoding="utf-8")
    harness = (formal_dir / "sc_theta_formal.v").read_text(encoding="utf-8")
    assert "depth 110" in job
    assert "32'sd131072" in harness
    assert "assert ($signed(theta_out) >= -32'sd205888);" in harness
    assert "assert ($signed(theta_out) <= 32'sd203817);" in harness
    assert "if (spike_out)" in harness
    assert "assert ($signed(theta_out) < 32'sd0);" in harness
