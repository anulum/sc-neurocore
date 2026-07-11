# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ermentrout-Kopell theta-Euler Python-to-Verilog co-simulation

"""Bounded fixed-point evidence for the maintained theta-Euler phase map."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float]


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
    """Return the generated RTL's committed event/phase trace."""
    assert HAS_IVERILOG, "iverilog is required for theta-Euler co-simulation"
    neuron = UniversalNeuron.from_schema("ermentrout_kopell_map_neuron")
    module_name = f"sc_ermentrout_kopell_q{data_width - fraction}_{fraction}_trace"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    current_q = Q88(data_width=data_width, fraction=fraction).encode(current)
    if current_q >= (1 << (data_width - 1)):
        current_q -= 1 << data_width
    current_literal = _signed_literal(data_width, current_q)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_ermentrout_kopell_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            f"wire signed [{data_width - 1}:0] theta_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t({current_literal}),",
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
            '        $display("EK_TRACE %0d %0d", spike_out, uut.theta_reg);',
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

    rows = re.findall(r"^EK_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << fraction)
    trace = [(int(event), int(theta_q) / scale) for event, theta_q in rows]
    assert len(trace) == n_steps, (
        f"theta-Euler RTL emitted {len(trace)} rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _hand_schema_trace(*, current: float, n_steps: int) -> list[TraceRow]:
    """Return exact hand/TOML/JSON evidence for one constant-current protocol."""
    hand = ErmentroutKopellMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "ermentrout_kopell_map_neuron.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "ermentrout_kopell_map_neuron.json")
    trace: list[TraceRow] = []

    for _step in range(n_steps):
        hand_event = hand.step(current)
        assert int(bool(toml_schema.step(I=current))) == hand_event
        assert int(bool(json_schema.step(I=current))) == hand_event
        assert toml_schema.state["theta"] == hand.theta
        assert json_schema.state["theta"] == hand.theta
        trace.append((hand_event, hand.theta))

    return trace


def _max_circular_error(expected: list[TraceRow], observed: list[TraceRow]) -> float:
    """Return maximum phase distance on the 2*pi circle."""
    two_pi = 2.0 * math.pi
    errors: list[float] = []
    for expected_row, observed_row in zip(expected, observed, strict=True):
        raw_error = abs(expected_row[1] - observed_row[1])
        errors.append(min(raw_error, abs(two_pi - raw_error)))
    return max(errors)


@pytest.mark.parametrize(
    ("current", "expected_events", "phase_tolerance"),
    (
        (-0.5, 0, 0.081),
        (0.5, 45, 0.089),
        (1.0, 64, 0.025),
    ),
    ids=("negative-drive", "intermediate-drive", "unit-drive"),
)
@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_q1616_class_correct_spike_count_and_circular_phase_bound(
    current: float,
    expected_events: int,
    phase_tolerance: float,
) -> None:
    """Q16.16 preserves spike counts with an explicit circular phase-error bound."""
    n_steps = 2000
    hand_trace = _hand_schema_trace(current=current, n_steps=n_steps)
    rtl_trace = _rtl_trace(
        n_steps=n_steps,
        current=current,
        data_width=32,
        fraction=16,
    )

    assert sum(row[0] for row in hand_trace) == expected_events
    assert sum(row[0] for row in rtl_trace) == expected_events
    assert _max_circular_error(hand_trace, rtl_trace) < phase_tolerance


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_negative_backward_wrap_is_event_silent_in_all_representations() -> None:
    """A first-step wrap from below zero to near 2*pi must emit no event."""
    hand_trace = _hand_schema_trace(current=-0.5, n_steps=20)
    rtl_trace = _rtl_trace(n_steps=20, current=-0.5, data_width=32, fraction=16)

    assert hand_trace[0][0] == rtl_trace[0][0] == 0
    assert hand_trace[0][1] > math.pi
    assert rtl_trace[0][1] > math.pi
    assert all(row[0] == 0 for row in hand_trace)
    assert all(row[0] == 0 for row in rtl_trace)


def test_hand_and_both_schema_formats_match_under_varied_drive() -> None:
    """TOML and JSON schemas match the maintained hand recurrence step-for-step."""
    hand = ErmentroutKopellMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "ermentrout_kopell_map_neuron.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIR / "ermentrout_kopell_map_neuron.json")
    currents = (-0.5, 0.0, 0.05, 0.1, 0.5, 1.0, 0.25, -0.1)

    for current in currents:
        for _step in range(250):
            hand_event = hand.step(current)
            assert int(bool(toml_schema.step(I=current))) == hand_event
            assert int(bool(json_schema.step(I=current))) == hand_event
            assert toml_schema.state["theta"] == hand.theta
            assert json_schema.state["theta"] == hand.theta
