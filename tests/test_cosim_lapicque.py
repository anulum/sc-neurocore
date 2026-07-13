# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque Python-to-Verilog fidelity contracts

"""Exact-flow paired-schema and Q16.16 RTL parity for Lapicque."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_Q1616_CASES = ((0.333, 0), (2.3, 83), (20.25, 500))


def _q1616_rtl_trace(current: float, n_steps: int) -> list[tuple[int, float]]:
    """Return the generated RTL's committed event and voltage trace."""
    neuron = UniversalNeuron.from_schema("lapicque")
    module_name = "sc_lapicque_exact_flow_q1616"
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
            '        $display("LAPICQUE_TRACE %0d %0d", spike_out, uut.v_reg);',
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
    rows = re.findall(r"^LAPICQUE_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << 16)
    trace = [(int(event), int(voltage_q) / scale) for event, voltage_q in rows]
    assert len(trace) == n_steps
    return trace


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_match_the_hand_exact_flow() -> None:
    """TOML and JSON preserve the varied exact-flow sequence and events."""
    hand = LapicqueNeuron()
    toml_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "lapicque.toml")
    json_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "lapicque.json")
    currents = (0.0, 0.3, 0.7, 1.1, 2.3, 5.7) * 200
    events = 0
    max_error = 0.0
    for current in currents:
        hand_event = hand.step(current)
        events += hand_event
        assert int(bool(toml_schema.step(I=current))) == hand_event
        assert int(bool(json_schema.step(I=current))) == hand_event
        max_error = max(
            max_error,
            abs(toml_schema.state["v"] - hand.v),
            abs(json_schema.state["v"] - hand.v),
        )
    assert events > 0
    assert max_error <= 2.0e-15


@pytest.mark.parametrize(
    ("current", "expected_events"),
    _Q1616_CASES,
    ids=[f"I={current:g}" for current, _events in _Q1616_CASES],
)
def test_q1616_preserves_event_vectors_and_voltage_bound(
    current: float,
    expected_events: int,
) -> None:
    """Preserve exact event timing and a measured Q16.16 voltage envelope.

    ``I=0.333`` is not exactly representable in Q16.16 and therefore exercises
    input quantisation. The driven points span silence, regular firing, and a
    high-rate train. The shared exponential look-up table bounds voltage error
    below 0.04 while every enrolled event remains cycle-exact.
    """
    n_steps = 1_000
    hand = LapicqueNeuron()
    schema = UniversalNeuron.from_schema("lapicque")
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
        < 0.04
    )
