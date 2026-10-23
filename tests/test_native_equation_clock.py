# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation clock and macro cadence tests

"""Observe real native state/RNG commit clocks, repeated fill and asynchronous reset."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from sc_neurocore.compiler.native_equation.clock import build_clock
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize(
    ("pipeline_registers", "substeps"),
    [(0, 1), (1, 1), (2, 1), (7, 1), (0, 3), (0, 5), (1 << 63, 1), (0, 1 << 64)],
)
def test_equation_clock_commits_state_and_macro_history_on_actual_edges(
    pipeline_registers: int, substeps: int, tmp_path: Path
) -> None:
    """State uses each substep; macro state waits; fill restarts after every recurrence."""
    b = ScGraphBuilder("equation_clock")
    clock = build_clock(b, pipeline_registers=pipeline_registers, substeps=substeps)
    assert clock.clocks_per_timestep == (pipeline_registers + 1 if pipeline_registers else substeps)
    state = b.state(b.constant_i64(0, "u16"), "u16")
    macro = b.state(b.constant_i64(0, "u64"), "u64")
    b.store(state, b.word_binary("add", state, b.constant_i64(1, "u16"), "u16"), clock.advance)
    b.store(
        macro, b.word_binary("add", macro, b.constant_i64(1, "u64"), "u64"), clock.macro_boundary
    )
    b.output("state_count", state)
    b.output("macro_count", macro)
    graph = b.build()
    assert graph.verify() is None
    graph = parse_ir(graph.to_text())
    count = state_count = macro_count = 0
    limit = pipeline_registers if pipeline_registers else substeps - 1
    expected = []
    stimulus = []
    for cycle in range(32):
        if cycle == 19:
            count = state_count = macro_count = 0
            stimulus.append(
                '#1; rst_n=0; #1; $display("OBS %h %h",state_count,macro_count); rst_n=1;'
            )
            expected.append("OBS 0000 0000000000000000")
        boundary = count == limit
        state_count += int(not pipeline_registers or boundary)
        macro_count += int(boundary)
        count = 0 if boundary else count + 1
        stimulus.append('#1; clk=1; #1; $display("OBS %h %h",state_count,macro_count); clk=0; #1;')
        expected.append(f"OBS {state_count:04x} {macro_count:016x}")
    bench = (
        "module tb; reg clk=0,rst_n=1; wire [15:0] state_count; wire [63:0] macro_count; "
        "equation_clock dut(.*); initial begin #1; rst_n=0; #1; rst_n=1;\n"
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, harness, executable = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(graph.emit_sv(), encoding="utf-8")
    harness.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(top), str(harness)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    observed = subprocess.run(
        ["vvp", str(executable)], capture_output=True, text=True, check=True, timeout=90
    ).stdout.splitlines()
    assert [line for line in observed if line.startswith("OBS ")] == expected
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"equation-clock-{pipeline_registers}-{substeps}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(graph.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(expected) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("pipeline_registers", "substeps"),
    [(-1, 1), (True, 1), (0, 0), (0, False), (1, 2), (1 << 64, 1), (0, (1 << 64) + 1)],
)
def test_equation_clock_refuses_invalid_cadence_before_graph_mutation(
    pipeline_registers: int, substeps: int
) -> None:
    """Malformed geometry must not leave counter storage or partial graph operations."""
    b = ScGraphBuilder("invalid_equation_clock")
    with pytest.raises(ValueError):
        build_clock(b, pipeline_registers=pipeline_registers, substeps=substeps)
    assert b.build().to_text() == ScGraphBuilder("invalid_equation_clock").build().to_text()
