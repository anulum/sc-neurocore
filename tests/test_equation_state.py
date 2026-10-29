# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Held equation state across native and generated targets

"""Execute authored held-state models through public compilation, replay and real RTL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS, EquationNeuron
from tests.engine_requirement import require_engine

require_engine()


@pytest.mark.parametrize("method", [*SUPPORTED_METHODS, "powers", "complete"])
def test_held_state_runs_in_native_registered_c_and_rust_targets(
    method: str,
    tmp_path: Path,
) -> None:
    """Held words survive integration, events and reset with identical target observations."""
    if method == "powers":
        model = EquationNeuron(
            {"v": "I**1.5+clip(I,.25,1)**clip(w,-1,1)"},
            state={"w": 0.5},
            method="map",
            dt=1.0,
        )
        held = "w"
        currents = [0.25, 0.5, 1.0, 0.75, 0.25, 0.5]
    elif method == "complete":
        model = EquationNeuron({"v": "I - v"}, state={"v": 0.5}, dt=0.25)
        held = None
        currents = [0.25, -0.5, 0.75, 0.0, 1.0, 0.25]
    else:
        model = EquationNeuron(
            {"v": "held + I - 0.5*v"},
            state={"held": 0.5},
            dt=0.25,
            method=method,
            threshold="v > 0.5",
            reset={"held": "held + 0.25", "v": "-0.25"},
            spike_value="held + v",
        )
        held = "held"
        currents = [0.25, -0.5, 0.75, 0.0, 1.0, 0.25]
    initial = dict(model.initial_state)
    model.step(0.25)
    live = dict(model.state)
    native = compile_to_ir(model, "state_native", function_lowering="v2")
    assert recompile_ir(native.graph.to_text()).graph.to_text() == native.graph.to_text()
    source_payload = native.graph.source_payload
    assert source_payload is not None
    payload = json.loads(source_payload)
    assert payload["model"]["equations"] == list(map(list, model.equations.items()))
    assert payload["model"]["initial_state"] == list(map(list, initial.items()))
    assert model.state == live and model.initial_state == initial
    registered = compile_to_verilog(model, "state_registered", function_lowering="v2")
    folded = compile_to_datapath(model, "state_folded", function_lowering="v2")
    outputs = dict(native.output_widths)
    assert held is None or f"{held}_out" in outputs
    source = tmp_path / "targets.sv"
    source.write_text(native.graph.emit_sv() + "\n" + registered + "\n" + folded)
    declarations = " ".join(
        f"wire [{bits - 1}:0] n_{name},r_{name};" for name, bits in outputs.items()
    )
    connections = {
        prefix: ",".join(
            [
                ".clk(clk)",
                ".rst_n(rst_n)",
                ".I_t(I_t)",
                *(f".{name}({prefix}_{name})" for name in outputs),
            ]
        )
        for prefix in ("n", "r")
    }
    comparison = " || ".join(f"n_{name} !== r_{name}" for name in outputs)
    state_names = [
        name.removesuffix("_out") for name in outputs if name in {f"{key}_out" for key in initial}
    ]
    declarations += " ".join(
        f"reg [15:0] p_{name}={round(initial[name] * 256) & 65535};" for name in state_names
    )
    declarations += " ".join(
        f"wire [{bits - 1}:0] p_{name};reg [{bits - 1}:0] expected_{name};"
        for name, bits in outputs.items()
    )
    folded_ports = [".I_t(I_t)", *(f".{name}_reg(p_{name})" for name in state_names)]
    folded_ports += [
        f".{name.removesuffix('_out')}_next_out(p_{name})"
        if name.removesuffix("_out") in state_names
        else f".{name}(p_{name})"
        for name in outputs
    ]
    folded_connection = "state_folded p(" + ",".join(folded_ports) + ");"
    capture_folded = "".join(f"expected_{name}=p_{name};" for name in outputs)
    advance_folded = "".join(f"p_{name}=expected_{name}_out;" for name in state_names)
    reset_folded = "".join(
        f"p_{name}=16'h{round(initial[name] * 256) & 65535:x};" for name in state_names
    )
    comparison += " || " + " || ".join(f"n_{name} !== expected_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    steps = []
    for index, current in enumerate(currents):
        reset = "rst_n=0;" + reset_folded + "#1;rst_n=1;#1;" if index == 3 else ""
        steps.append(
            f'{reset}I_t=16\'h{round(current * 256) & 65535:x};#2;{capture_folded}clk=1;#1;if({comparison}) $fatal(1,"state mismatch");{display}clk=0;{advance_folded}#1;'
        )
    bench = (
        f"module tb;reg clk=0,rst_n=1;reg [15:0] I_t=0;{declarations}"
        f"state_native n({connections['n']});state_registered r({connections['r']});{folded_connection}"
        "initial begin #1;rst_n=0;#1;rst_n=1;" + "\n".join(steps) + "$finish;end endmodule\n"
    )
    tb, binary = tmp_path / "tb.sv", tmp_path / "run"
    tb.write_text(bench)
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    observed = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    rows = [line.split()[1:] for line in observed.splitlines() if line.startswith("OBS ")]
    assert len(rows) == len(currents)
    assert all(not ({"x", "z"} & set(word.lower())) for row in rows for word in row)
    if held is not None and method == "powers":
        held_index = list(outputs).index(f"{held}_out")
        assert all(int(row[held_index], 16) == 128 for row in rows)
    for language in ("c", "rust"):
        code = generate_bittrue_kernel_from_neuron(
            model, module_name="state_kernel", language=language, function_lowering="v2"
        )
        names = list(outputs)
        if language == "c":
            code += "\n#include <stdio.h>\nint main(void){state_kernel_state_t s;state_kernel_reset(&s);\n"
            for index, current in enumerate(currents):
                code += "state_kernel_reset(&s);\n" if index == 3 else ""
                code += (
                    f'state_kernel_step(&s,{round(current * 256)});printf("OBS '
                    + " ".join("%x" for _ in names)
                    + '\\n",'
                    + ",".join(
                        f"(unsigned)(s.{name}) & " + ("1" if outputs[name] == 1 else "65535")
                        for name in names
                    )
                    + ");\n"
                )
            code += "return 0;}\n"
            program = tmp_path / "kernel.c"
            program.write_text(code)
            command = [
                "cc",
                "-std=c11",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                str(program),
                "-o",
                str(tmp_path / "kernel-c"),
            ]
        else:
            fields = list(model.equations) + ([held] if held is not None else [])
            fields = list(dict.fromkeys(fields))
            initialisers = [*(f"{name}:0,{name}_out:0" for name in fields), "spike_out:0"]
            if "spike_value_out" in names:
                initialisers.append("spike_value_out:0")
            code += (
                "\nfn main(){let mut s=State_kernelState{"
                + ",".join(initialisers)
                + "};s.reset();\n"
            )
            for index, current in enumerate(currents):
                code += "s.reset();\n" if index == 3 else ""
                code += (
                    f's.step({round(current * 256)});println!("OBS '
                    + " ".join("{:x}" for _ in names)
                    + '",'
                    + ",".join(
                        f"(s.{name} as u64) & " + ("1" if outputs[name] == 1 else "65535")
                        for name in names
                    )
                    + ");\n"
                )
            code += "}\n"
            program = tmp_path / "kernel.rs"
            program.write_text(code)
            command = [
                "rustc",
                "--edition=2021",
                "-C",
                "overflow-checks=on",
                str(program),
                "-o",
                str(tmp_path / "kernel-rust"),
            ]
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        run = subprocess.run(
            [command[-1]], capture_output=True, text=True, check=True, timeout=60
        ).stdout
        actual = [line.split()[1:] for line in run.splitlines() if line.startswith("OBS ")]
        assert [[int(word, 16) for word in row] for row in actual] == [
            [int(word, 16) for word in row] for row in rows
        ]
    assert model.get_state() == live and model.initial_state == initial
    if held is not None:
        model.reset()
        held_index = list(outputs).index(f"{held}_out")
        for index, current in enumerate(currents):
            if index == 3:
                model.reset()
            model.step(current)
            assert int(rows[index][held_index], 16) == round(model.state[held] * 256) & 65535
    if receipt_root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        receipt = Path(receipt_root) / f"held-state-{method}"
        receipt.mkdir(parents=True, exist_ok=True)
        (receipt / "graph.scir").write_text(native.graph.to_text())
        (receipt / "graph.mlir").write_text(native.graph.emit_mlir())
        (receipt / "tb.sv").write_text(bench + registered + folded)
        (receipt / "expected.txt").write_text(
            "\n".join(line for line in observed.splitlines() if line.startswith("OBS ")) + "\n"
        )


def test_held_numeric_state_cannot_be_called_as_a_namespace_function() -> None:
    """All maintained whole-model compilers refuse the same shadowed authored call."""
    model = EquationNeuron({"v": "exp(I)"}, state={"exp": 0.5}, method="map")
    with pytest.raises(ValueError, match="numeric"):
        compile_to_ir(model)
    with pytest.raises(ValueError, match="numeric"):
        compile_to_verilog(model)
    with pytest.raises(ValueError, match="numeric"):
        compile_to_datapath(model)
    for language in ("c", "rust"):
        with pytest.raises(ValueError, match="numeric"):
            generate_bittrue_kernel_from_neuron(model, language=language)


def test_compilation_preserves_held_state_and_the_running_random_sequence() -> None:
    """Every maintained compiler leaves the next real model draw and reset unchanged."""
    model = EquationNeuron(
        {"v": "held + rand() * I"},
        state={"held": 0.5},
        method="map",
        dt=1.0,
        rng_seed=(1 << 63) + 7,
        random_stream=(1 << 31) + 3,
    )
    control = EquationNeuron(
        {"v": "held + rand() * I"},
        state={"held": 0.5},
        method="map",
        dt=1.0,
        rng_seed=(1 << 63) + 7,
        random_stream=(1 << 31) + 3,
    )
    model.step(0.25)
    control.step(0.25)
    live = model.get_state()
    initial = dict(model.initial_state)
    first = compile_to_ir(model).graph.to_text()
    assert recompile_ir(first).graph.to_text() == first
    compile_to_verilog(model)
    compile_to_datapath(model)
    for language in ("c", "rust"):
        generate_bittrue_kernel_from_neuron(model, language=language)
    assert model.get_state() == live
    assert model.initial_state == initial
    model.step(0.75)
    control.step(0.75)
    assert model.get_state() == control.get_state()
    assert compile_to_ir(model).graph.to_text() == first
    model.reset()
    control.reset()
    assert model.get_state() == control.get_state() == initial
    model.step(0.5)
    control.step(0.5)
    assert model.get_state() == control.get_state()
