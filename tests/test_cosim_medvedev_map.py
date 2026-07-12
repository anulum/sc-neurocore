# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev map Python-to-Verilog co-simulation

"""Bounded Q16.16 evidence for the Medvedev first-return map."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float]


def _signed_literal(value: int) -> str:
    """Render one signed 32-bit testbench literal."""
    if value < 0:
        return f"-32'sd{abs(value)}"
    return f"32'sd{value}"


def _rtl_q1616_trace(n_steps: int, current: float) -> list[TraceRow]:
    """Compile and execute the generated Q16.16 RTL trace."""
    assert HAS_IVERILOG, "iverilog is required for Medvedev co-simulation"
    neuron = UniversalNeuron.from_schema("medvedev_map")
    module_name = "sc_medvedev_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_medvedev_map_q1616_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] u_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t({_signed_literal(current_q)}),",
            "    .spike_out(spike_out), .u_out(u_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            (
                f"    for (step_index = 0; step_index < {n_steps}; "
                "step_index = step_index + 1) begin"
            ),
            "        @(posedge clk); #1;",
            '        $display("MEDVEDEV_TRACE %0d %0d", spike_out, u_out);',
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

    rows = re.findall(r"^MEDVEDEV_TRACE (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    scale = float(1 << 16)
    trace = [(int(event), int(u_q) / scale) for event, u_q in rows]
    assert len(trace) == n_steps, (
        f"Medvedev RTL emitted {len(trace)} rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _hand_schema_trace(current: float, n_steps: int) -> list[TraceRow]:
    """Return exact hand/TOML/JSON evidence for the calibrated recurrence."""
    hand = MedvedevMapNeuron()
    toml_schema = UniversalNeuron.from_schema(SCHEMA_DIRECTORY / "medvedev_map.toml")
    json_schema = UniversalNeuron.from_schema(SCHEMA_DIRECTORY / "medvedev_map.json")
    trace: list[TraceRow] = []
    for _step in range(n_steps):
        hand_event = hand.step(current)
        toml_event = int(bool(toml_schema.step(I=current)))
        json_event = int(bool(json_schema.step(I=current)))
        assert toml_event == json_event == hand_event
        assert toml_schema.state == json_schema.state == {"u": hand.u}
        trace.append((hand_event, hand.u))
    return trace


def test_hand_toml_and_json_are_exact_at_enrolled_operating_point() -> None:
    """Both authored schemas must equal the hand recurrence exactly."""
    trace = _hand_schema_trace(current=2.0, n_steps=100)
    assert sum(row[0] for row in trace) == 75


def test_q1616_event_vector_and_state_envelope() -> None:
    """Q16.16 must retain all events and the bounded slow-calcium orbit."""
    hand_trace = _hand_schema_trace(current=2.0, n_steps=100)
    rtl_trace = _rtl_q1616_trace(n_steps=100, current=2.0)
    assert [row[0] for row in rtl_trace] == [row[0] for row in hand_trace]
    assert sum(row[0] for row in rtl_trace) == 75
    state_error = max(
        abs(expected[1] - observed[1])
        for expected, observed in zip(hand_trace, rtl_trace, strict=True)
    )
    assert state_error < 0.007813


def test_q1616_is_structurally_required_by_calibrated_d() -> None:
    """The disclosed Eq. 4.13 scale fits Q16.16 but not signed Q8.8."""
    d = MedvedevMapNeuron().d
    q88 = Q88(data_width=16, fraction=8)
    q1616 = Q88(data_width=32, fraction=16)
    assert d > q88.max_value
    assert d < q1616.max_value
    assert q88.check_range(d, label="d")
    assert q1616.check_range(d, label="d") == []
