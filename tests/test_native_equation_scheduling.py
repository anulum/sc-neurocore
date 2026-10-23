# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation pipeline scheduling parity

"""Compare selected native product storage with the public registered RTL compiler."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.native_equation.clock import build_clock
from sc_neurocore.compiler.native_equation.integrators import integrate
from sc_neurocore.compiler.native_equation.scheduling import EquationSchedule
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("method", ["map", "euler", "gauss_seidel"])
@pytest.mark.parametrize(
    ("expression_name", "expression"),
    [
        ("division", "(v / 2) + (v * 0.5) + (I * 0.25)"),
        ("clamp", "min(abs(v * 0.5), I * 0.25) + 0.125"),
        ("power", "((v ** 2) / 2) + (I * 0.25)"),
    ],
)
@pytest.mark.parametrize(("width", "fraction"), [(16, 8), (64, 8)])
@pytest.mark.parametrize("selection", ["all", "mul1", "_mul2", "_dt_mul_v"])
def test_native_pipeline_matches_every_state_clock_and_async_reset(
    method: str,
    expression_name: str,
    expression: str,
    width: int,
    fraction: int,
    selection: str,
    tmp_path: Path,
) -> None:
    """Preserve numbering after divisions/powers, shared clamp nodes and repeated fill."""
    neuron = EquationNeuron(
        {"v": expression, "w": "(v * 0.25) - (w * 0.5)"},
        state={"v": -0.5, "w": 0.75},
        dt=0.5,
        method=method,
    )
    q = Q88(width, fraction, overflow="wrap", rounding="bankers")
    b = ScGraphBuilder("native_pipeline")
    a = EquationArithmetic(b, q)
    states = {
        name: b.state(a.literal(neuron.initial_state[name], width), f"i{width}")
        for name in neuron.equations
    }
    current = b.input("I_t", f"i{width}")
    schedule = EquationSchedule(
        q,
        b.word_delay,
        pipeline=selection == "all",
        points=() if selection == "all" else (selection,),
    )

    def commit(value: int, raw_width: int) -> int:
        """Apply the declared wrap policy to an integration candidate."""
        return b.word_resize(value, f"i{width}")

    candidates = integrate(neuron, a, states, {"I": current}, commit, schedule=schedule)
    clock = build_clock(b, pipeline_registers=schedule.register_count)
    for name, state in states.items():
        b.store(state, candidates[name], clock.advance)
        b.output(f"{name}_out", state)
    graph = b.build()
    assert graph.verify() is None
    graph = parse_ir(graph.to_text())
    legacy = compile_to_verilog(
        neuron,
        "legacy_pipeline",
        width,
        fraction,
        overflow="wrap",
        rounding="bankers",
        pipeline_stages=int(selection == "all"),
        pipeline_points=[] if selection == "all" else [selection],
    )
    latency_port = ""
    latency_assertion = ""
    latency_declaration = ""
    if schedule.register_count:
        latency_port = ",.latency(legacy_latency)"
        latency_declaration = "wire [31:0] legacy_latency; "
        latency_assertion = (
            f"if (legacy_latency !== 32'd{schedule.register_count}) "
            '$fatal(1,"product register count differs"); '
        )
    stimulus = []
    samples = [0, 128, -128, 1, -1, 256, -256, 31]
    for cycle in range(64):
        if cycle in (19, 43):
            stimulus.append(
                "#1; rst_n=0; #1; if (nv !== lv || nw !== lw) "
                '$fatal(1,"pipeline asynchronous reset mismatch"); rst_n=1;'
            )
        code = samples[cycle % len(samples)] & ((1 << width) - 1)
        stimulus.append(
            f"I_t={width}'h{code:x}; #2; clk=1; #1; "
            "if (nv !== lv || nw !== lw) "
            '$fatal(1,"pipeline state mismatch: %h %h / %h %h",nv,lv,nw,lw); '
            '$display("OBS %h %h",nv,nw); clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg signed [{width - 1}:0] I_t=0; "
        f"wire [{width - 1}:0] nv,nw,lv,lw; "
        + latency_declaration
        + "native_pipeline native_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(nv),.w_out(nw)); "
        + "legacy_pipeline legacy_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(lv),.w_out(lw),.spike_out()"
        + latency_port
        + "); initial begin #1; rst_n=0; #1; rst_n=1; #1; "
        + latency_assertion
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, harness, executable = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(graph.emit_sv() + "\n" + legacy, encoding="utf-8")
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
    observations = [line for line in observed if line.startswith("OBS ")]
    assert len(observations) == len(samples) * 8
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"pipeline-{method}-{width}-{fraction}-{selection}-{expression_name}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(graph.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench + legacy, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
