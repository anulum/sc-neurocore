# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native neuron transaction RTL parity

"""Exercise native state/event/clock wiring against the public registered compiler."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import cast

import pytest

from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.native_equation.commits import EquationCommits
from sc_neurocore.compiler.native_equation.transactions import lower_neuron
from sc_neurocore.compiler import compile_to_ir
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS, EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize(
    ("stages", "points", "error"),
    [
        (True, [], TypeError),
        (1.5, [], TypeError),
        ("1", [], TypeError),
        (None, [], TypeError),
        (-1, [], ValueError),
        (0, "mul1", TypeError),
        (0, [1], TypeError),
        (0, ["mul1", "mul1"], ValueError),
        (1, ["mul1"], ValueError),
    ],
)
def test_native_neuron_pipeline_refusals_match_public_compiler(
    stages: object, points: object, error: type[Exception]
) -> None:
    """Refuse malformed pipeline requests through both actual compiler entry points."""
    neuron = EquationNeuron({"v": "I - v"}, dt=0.5)
    b = ScGraphBuilder("invalid_pipeline")
    arithmetic = EquationArithmetic(b, Q88(16, 8, overflow="wrap"))

    def commit(raw: int, _width: int) -> int:
        """Narrow a state word under the declared wrap policy."""
        return b.word_resize(raw, "i16")

    with pytest.raises(error):
        compile_to_verilog(
            neuron,
            overflow="wrap",
            pipeline_stages=cast(int, stages),
            pipeline_points=cast(list[str], points),
        )
    with pytest.raises(error):
        lower_neuron(
            neuron,
            arithmetic,
            commit,
            pipeline_stages=cast(int, stages),
            pipeline_points=cast(list[str], points),
        )


@pytest.mark.parametrize(("signed", "rounding"), [(False, "truncate"), (True, "stochastic")])
def test_native_transaction_numeric_refusal_preserves_the_real_builder(
    signed: bool, rounding: str
) -> None:
    """Standalone transaction admission matches the public compiler and appends no invalid model."""
    neuron = EquationNeuron({"v": "I - v"}, dt=0.5)
    builder = ScGraphBuilder("numeric_refusal")
    invalid = EquationArithmetic(builder, Q88(16, 8, signed=signed, rounding=rounding))

    def wrap(raw: int, _width: int) -> int:
        """Supply the standalone API's actual signed-word narrowing callback."""
        return builder.word_resize(raw, "i16")

    with pytest.raises(NotImplementedError):
        lower_neuron(neuron, invalid, wrap)
    with pytest.raises(NotImplementedError):
        compile_to_verilog(neuron, signed=signed, rounding=rounding)
    arithmetic = EquationArithmetic(builder, Q88(16, 8))
    policy = EquationCommits(arithmetic)
    lower_neuron(neuron, arithmetic, policy.state, event_commit=policy.event)
    actual = builder.build()
    fresh = ScGraphBuilder("numeric_refusal")
    other = EquationArithmetic(fresh, Q88(16, 8))
    clean_policy = EquationCommits(other)
    lower_neuron(neuron, other, clean_policy.state, event_commit=clean_policy.event)
    assert actual.verify() is None
    assert actual.to_text() == fresh.build().to_text()


@pytest.mark.parametrize("case", ["state_port", "macro_level", "underflow", "stochastic_pipeline"])
def test_public_transaction_refuses_invalid_model_or_cadence(case: str) -> None:
    """Both real compiler entry points reject unsupported model/cadence combinations."""
    error: type[Exception]
    stages = 0
    if case == "state_port":
        neuron = EquationNeuron({"spike": "I"}, dt=0.5)
        error, message = ValueError, "collides with a spike output"
    elif case == "macro_level":
        neuron = EquationNeuron({"v": "I"}, substeps=3, dt=0.5)
        error, message = NotImplementedError, "macro substeps"
    elif case == "underflow":
        neuron = EquationNeuron({"v": "I - v"}, dt=1e-9)
        error, message = ValueError, "dt underflows"
    else:
        neuron = EquationNeuron(
            {"v": "gain * v + I"},
            parameters={"gain": 0.5},
            reset={"v": "0"},
            detection="poisson",
            probability_expression="clip(v, 0, 1)",
            method="map",
            dt=1.0,
        )
        stages = 1
        error, message = NotImplementedError, "stochastic thresholds"
    with pytest.raises(error, match=message):
        compile_to_ir(neuron, pipeline_stages=stages)
    with pytest.raises(error):
        compile_to_verilog(neuron, pipeline_stages=stages)


def _neuron(case: str) -> EquationNeuron:
    if case in SUPPORTED_METHODS:
        return EquationNeuron(
            {"v": "gain * (I - v + w)", "w": "(v - w) / tau"},
            state={"v": -0.5, "w": 0.75},
            parameters={"gain": 0.25},
            constants={"tau": 8.0},
            threshold="v > 2",
            reset={"v": "w", "w": "v_prev"},
            spike_value="v - w_prev",
            method=case,
            dt=0.5,
        )
    if case == "zero_dt":
        return EquationNeuron({"v": "I - v"}, state={"v": 0.5}, dt=0.0, threshold="v > 1")
    if case == "macro_partial":
        return EquationNeuron(
            {"v": "I", "w": "w + I / 2"},
            state={"v": 0.25, "w": -0.5},
            method="map",
            dt=1.0,
            threshold="v > v_prev",
            spike_value="v - v_prev",
            detection="crossing",
            substeps=3,
        )
    if case == "macro":
        return EquationNeuron(
            {"v": "I"},
            state={"v": 0.25},
            method="map",
            dt=1.0,
            threshold="v > v_prev",
            spike_value="v - v_prev",
            detection="crossing",
            substeps=3,
        )
    if case == "draws":
        return EquationNeuron(
            {"v": "rand() * gain + I"},
            parameters={"gain": 0.5},
            state={"v": -0.5},
            method="map",
            dt=1.0,
            threshold="v > 0.75",
            reset={"v": "0"},
            rng_seed=(1 << 63) + 1234,
            random_stream=(1 << 31) + 2,
        )
    if case == "diffusion":
        return EquationNeuron(
            {"v": "-v / 2 + xi + I"},
            state={"v": 0.0},
            dt=0.5,
            threshold="v > 1",
            reset={"v": "0"},
            rng_seed=1234,
        )
    if case in {"poisson", "philox", "escape"}:
        return EquationNeuron(
            {"v": "v + I"},
            state={"v": 0.0},
            reset={"v": "0"},
            method="map",
            dt=0.5,
            rng_seed=1234,
            detection="escape_rate" if case == "escape" else "poisson",
            rate_expression="abs(v)" if case == "escape" else None,
            probability_expression=None if case == "escape" else "clip(v, 0, 1)",
            threshold_rng="philox" if case == "philox" else "lfsr16",
        )
    raise ValueError(f"unknown transaction case: {case}")


@pytest.mark.parametrize(
    ("case", "pipeline"),
    [
        (name, enabled)
        for name in [
            *SUPPORTED_METHODS,
            "macro",
            "macro_partial",
            "zero_dt",
            "draws",
            "diffusion",
            "poisson",
            "philox",
            "escape",
        ]
        for enabled in (
            (False,) if name in {"macro", "macro_partial", "poisson", "escape"} else (False, True)
        )
    ],
)
@pytest.mark.parametrize("lowering", ["lut-v1", "v2"])
def test_native_neuron_transaction_preserves_public_state_spikes_and_reset(
    case: str, lowering: str, pipeline: bool, tmp_path: Path
) -> None:
    """Observe every output through initial state, recurrent fill, RNG and macro boundaries."""
    neuron = _neuron(case)
    stages = int(pipeline)
    q = Q88(16, 8, overflow="saturate", rounding="bankers", function_lowering=lowering)
    transaction = compile_to_ir(
        neuron,
        "native_transaction",
        16,
        8,
        rounding="bankers",
        function_lowering=lowering,
        pipeline_stages=stages,
    )
    graph = transaction.graph
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    legacy = compile_to_verilog(
        neuron,
        "legacy_transaction",
        16,
        8,
        rounding="bankers",
        function_lowering=lowering,
        pipeline_stages=stages,
    )
    outputs = dict(transaction.output_widths)
    declarations = " ".join(
        f"wire [{width - 1}:0] n_{name},l_{name};" for name, width in outputs.items()
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
        for prefix in ("n", "l")
    }
    width_check = ""
    if transaction.pipeline_registers:
        bits = transaction.pipeline_registers.bit_length()
        width_check = (
            f"if ($bits(native_dut.latency) != {bits} || "
            f"$bits(legacy_dut.latency) != {bits}) "
            '$fatal(1,"public latency port width mismatch"); '
        )
    comparisons = " || ".join(f"n_{name} !== l_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    samples = [0.0, 0.5, -0.25, 1.0, -1.0, 0.125, 2.0, -2.0]
    stimulus = []
    for cycle in range(64):
        if cycle in (19, 43):
            stimulus.append(
                f'#1; rst_n=0; #1; if ({comparisons}) $fatal(1,"transaction reset mismatch"); rst_n=1;'
            )
        value = q.encode(samples[cycle % len(samples)])
        stimulus.append(
            f'I_t=16\'h{value:04x}; #2; clk=1; #1; if ({comparisons}) $fatal(1,"transaction outputs mismatch"); {display} clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg [15:0] I_t=0; {declarations} "
        f"native_transaction native_dut({connections['n']}); "
        f"legacy_transaction legacy_dut({connections['l']}); "
        f'initial begin {width_check} #1; rst_n=0; #1; if ({comparisons}) $fatal(1,"initial transaction mismatch"); rst_n=1;\n'
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, harness, executable = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(restored.emit_sv() + "\n" + legacy, encoding="utf-8")
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
    assert len(observations) == 64
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        directory = Path(root) / f"transaction-{case}-{lowering}-{pipeline}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "graph.scir").write_text(restored.to_text(), encoding="utf-8")
        (directory / "graph.mlir").write_text(restored.emit_mlir(), encoding="utf-8")
        (directory / "tb.sv").write_text(bench + legacy, encoding="utf-8")
        (directory / "expected.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
