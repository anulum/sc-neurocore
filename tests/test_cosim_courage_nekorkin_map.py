# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Courbage-Nekorkin map Python-to-Verilog co-simulation

"""Bounded fixed-point trajectory evidence for the Courbage-Nekorkin map."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float, float]


def _signed_literal(width: int, value: int) -> str:
    """Return a signed Verilog literal with the sign outside the width token."""
    if value < 0:
        return f"-{width}'sd{-value}"
    return f"{width}'sd{value}"


def _rtl_trace(
    *,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
) -> list[TraceRow]:
    """Return the generated RTL's committed event and two-state trace."""
    assert HAS_IVERILOG, "iverilog is required for Courbage-Nekorkin co-simulation"
    neuron = UniversalNeuron.from_schema("courage_nekorkin_map")
    module_name = f"sc_courbage_nekorkin_q{data_width - fraction}_{fraction}_trace"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    current_q = Q88(data_width=data_width, fraction=fraction).encode(current)
    current_literal = _signed_literal(data_width, current_q)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_courbage_nekorkin_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            f"wire signed [{data_width - 1}:0] x_out;",
            f"wire signed [{data_width - 1}:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t({current_literal}),",
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
            ('        $display("COURBAGE_TRACE %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);'),
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
        r"^COURBAGE_TRACE (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    scale = float(1 << fraction)
    trace = [(int(event), int(x_q) / scale, int(y_q) / scale) for event, x_q, y_q in rows]
    assert len(trace) == n_steps, (
        f"Courbage-Nekorkin RTL emitted {len(trace)} rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _hand_schema_trace(
    *,
    current: float,
    n_steps: int,
) -> tuple[list[TraceRow], dict[str, int]]:
    """Return exact hand/schema evidence and published branch counts."""
    hand = CourageNekorkinMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "courage_nekorkin_map.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "courage_nekorkin_map.json")
    trace: list[TraceRow] = []
    branch_counts = {
        "low": 0,
        "middle": 0,
        "high": 0,
        "heaviside0": 0,
        "heaviside1": 0,
    }

    for _step in range(n_steps):
        j_min, j_max = hand._breakpoints()
        branch = "low" if hand.x <= j_min else "middle" if hand.x < j_max else "high"
        branch_counts[branch] += 1
        branch_counts["heaviside1" if hand.x >= hand.d else "heaviside0"] += 1

        hand_event = hand.step(current)
        assert int(bool(toml_schema.step(I=current))) == hand_event
        assert int(bool(json_schema.step(I=current))) == hand_event
        assert toml_schema.state == {"x": hand.x, "y": hand.y}
        assert json_schema.state == {"x": hand.x, "y": hand.y}
        trace.append((hand_event, hand.x, hand.y))

    return trace, branch_counts


def _state_errors(expected: list[TraceRow], observed: list[TraceRow]) -> tuple[float, float]:
    """Return maximum absolute errors for x and y over two aligned traces."""
    x_error = max(
        abs(expected_row[1] - observed_row[1])
        for expected_row, observed_row in zip(expected, observed, strict=True)
    )
    y_error = max(
        abs(expected_row[2] - observed_row[2])
        for expected_row, observed_row in zip(expected, observed, strict=True)
    )
    return x_error, y_error


@pytest.mark.parametrize(
    (
        "current",
        "n_steps",
        "expected_events",
        "expected_branches",
        "x_tolerance",
        "y_tolerance",
    ),
    (
        (
            -0.3,
            30,
            1,
            {"low": 22, "middle": 1, "high": 7, "heaviside0": 23, "heaviside1": 7},
            0.002,
            0.00021,
        ),
        (
            0.0,
            20,
            2,
            {"low": 13, "middle": 7, "high": 0, "heaviside0": 16, "heaviside1": 4},
            0.014,
            0.00031,
        ),
        (
            0.3,
            30,
            1,
            {"low": 7, "middle": 1, "high": 22, "heaviside0": 8, "heaviside1": 22},
            0.0005,
            0.00006,
        ),
    ),
    ids=("negative-drive", "autonomous", "positive-drive"),
)
def test_q1616_bounded_trajectory(
    current: float,
    n_steps: int,
    expected_events: int,
    expected_branches: dict[str, int],
    x_tolerance: float,
    y_tolerance: float,
) -> None:
    """Q16.16 must preserve the bounded event vector and branch semantics."""
    hand_trace, branch_counts = _hand_schema_trace(current=current, n_steps=n_steps)
    rtl_trace = _rtl_trace(
        n_steps=n_steps,
        current=current,
        data_width=32,
        fraction=16,
    )

    assert branch_counts == expected_branches
    assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
    assert sum(row[0] for row in rtl_trace) == expected_events
    x_error, y_error = _state_errors(hand_trace, rtl_trace)
    assert x_error < x_tolerance
    assert y_error < y_tolerance


@pytest.mark.parametrize(
    ("current", "expected_events"),
    ((-0.3, 1), (0.0, 4), (0.3, 1)),
    ids=("negative-drive", "autonomous", "positive-drive"),
)
def test_q3232_short_window_trajectory(current: float, expected_events: int) -> None:
    """Q32.32 must preserve all 30-step events with sub-3e-5 state error."""
    n_steps = 30
    hand_trace, _branch_counts = _hand_schema_trace(current=current, n_steps=n_steps)
    rtl_trace = _rtl_trace(
        n_steps=n_steps,
        current=current,
        data_width=64,
        fraction=32,
    )

    assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
    assert sum(row[0] for row in rtl_trace) == expected_events
    x_error, y_error = _state_errors(hand_trace, rtl_trace)
    assert x_error < 0.00003
    assert y_error < 0.000001


def test_q1616_declares_autonomous_sensitive_boundary() -> None:
    """The autonomous 30-step Q16.16 trajectory must stay outside the parity band."""
    hand_trace, _branch_counts = _hand_schema_trace(current=0.0, n_steps=30)
    rtl_trace = _rtl_trace(
        n_steps=30,
        current=0.0,
        data_width=32,
        fraction=16,
    )
    event_mismatches = sum(
        expected[0] != observed[0] for expected, observed in zip(hand_trace, rtl_trace, strict=True)
    )

    assert sum(row[0] for row in hand_trace) == 4
    assert sum(row[0] for row in rtl_trace) == 6
    assert event_mismatches == 6
