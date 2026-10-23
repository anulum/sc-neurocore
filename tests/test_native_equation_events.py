# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation event RTL parity

"""Observe reset environments, crossing history and stochastic/graded spikes in real RTL."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_lowering import prepare_neuron
from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.native_equation.events import lower_events
from sc_neurocore.compiler.native_equation.integrators import integrate
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def _model(name: str) -> EquationNeuron:
    if name == "crossing":
        return EquationNeuron(
            {"v": "I"},
            state={"v": 1.0},
            threshold="v > 0",
            method="map",
            dt=1.0,
            detection="crossing",
        )
    if name == "reset":
        return EquationNeuron(
            {"v": "v + I", "w": "w + 1"},
            state={"v": 0.0, "w": -0.25},
            threshold="(v >= 2 and w <= 3) or v < -3",
            reset={"v": "w_prev - I", "w": "v + w"},
            spike_value="v - w_prev",
            method="map",
            dt=1.0,
        )
    if name == "escape":
        return EquationNeuron(
            {"v": "v + I"},
            state={"v": 0.0},
            reset={"v": "0"},
            detection="escape_rate",
            rate_expression="abs(v)",
            method="map",
            dt=0.5,
            rng_seed=1234,
        )
    if name in {"poisson", "philox"}:
        return EquationNeuron(
            {"v": "v + I"},
            state={"v": 0.0},
            reset={"v": "0"},
            detection="poisson",
            probability_expression="clip(v, 0, 1)",
            method="map",
            dt=1.0,
            rng_seed=1234,
            threshold_rng="philox" if name == "philox" else "lfsr16",
        )
    raise ValueError(f"unknown event model: {name}")


@pytest.mark.parametrize("model", ["reset", "crossing", "escape", "poisson", "philox"])
@pytest.mark.parametrize("lowering", ["lut-v1", "v2"])
def test_native_event_transaction_matches_public_compiler(
    model: str, lowering: str, tmp_path: Path
) -> None:
    """Preserve actual event traces, including zero/one/interior probability and mid-run reset."""
    original = _model(model)
    neuron = prepare_neuron(original)
    q = Q88(16, 8, overflow="saturate", function_lowering=lowering)
    b = ScGraphBuilder("native_events")
    a = EquationArithmetic(b, q)
    states = {
        name: b.state(a.literal(neuron.initial_state[name], 16), "i16") for name in neuron.equations
    }
    current = b.input("I_t", "i16")
    true = b.constant_i64(1, "bool")
    externs: dict[str, int] = {}
    if neuron.uses_random_draws:
        counter = b.state(a.constant(0, 64), "i64")
        b.store(counter, b.word_binary("add", counter, a.constant(1, 64), "i64"))
        externs = {
            "step": counter,
            "seed": a.constant(neuron.random_seed, 64),
            "stream": a.constant(neuron.random_stream, 64),
        }

    def commit(raw: int, width: int) -> int:
        return b.word_resize(b.word_clamp(raw, -32768, 32767, f"i{width}"), "i16")

    candidates = integrate(neuron, a, states, {"I": current}, commit, externs)
    event = lower_events(neuron, a, candidates, states, {"I": current}, commit, true, externs)
    outputs: dict[str, str] = {}
    for name in states:
        b.store(states[name], event.states[name])
        outputs[f"{name}_out"] = "[15:0]"
        b.output(f"{name}_out", states[name])
    spike = b.state(b.constant_i64(0, "bool"), "bool")
    b.store(spike, event.spike)
    b.output("spike_out", spike)
    outputs["spike_out"] = ""
    if event.value is not None:
        graded = b.state(a.constant(0, 16), "i16")
        b.store(graded, event.value)
        b.output("spike_value_out", graded)
        outputs["spike_value_out"] = "[15:0]"
    graph = b.build()
    assert graph.verify() is None
    graph = parse_ir(graph.to_text())
    legacy = compile_to_verilog(original, "legacy_events", 16, 8, function_lowering=lowering)
    declarations = " ".join(f"wire {word} n_{name},l_{name};" for name, word in outputs.items())
    connections = {
        prefix: ",".join(
            [
                ".clk(clk)",
                ".rst_n(rst_n)",
                ".I_t(I_t)",
                *(f".{name}({prefix}_{name})" for name in outputs),
            ]
        )
        for prefix in ("n", "l")
    }
    comparisons = " || ".join(f"n_{name} !== l_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    samples = [0.0, 0.0, -0.25, 0.25, 0.25, 0.0, 0.5, 1.0, -1.0, 2.0, -4.0, 0.0, 0.25, 1.0]
    stimulus = []
    for index, value in enumerate(samples):
        if index == 8:
            stimulus.append(
                f'#1; rst_n=0; #1; if ({comparisons}) $fatal(1,"reset mismatch"); rst_n=1;'
            )
        stimulus.append(
            f'I_t=16\'h{q.encode(value):04x}; #2; clk=1; #1; if ({comparisons}) $fatal(1,"native event mismatch"); {display} clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg [15:0] I_t=0; {declarations} native_events n_dut({connections['n']}); legacy_events l_dut({connections['l']}); initial begin #1; rst_n=0; #1; rst_n=1;\n"
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(graph.emit_sv() + "\n" + legacy, encoding="utf-8")
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    observed = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=90
    ).stdout.splitlines()
    observations = [line for line in observed if line.startswith("OBS ")]
    assert len(observations) == len(samples)
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"events-{model}-{lowering}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(graph.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench + legacy, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
