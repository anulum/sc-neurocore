# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discontinuous fixed-point function references
"""Execute discrete equations through public compilation, kernels and source replay."""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import pytest
from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from tests.bit_true_cosim_support import _testbench, _c_main, _rows
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine
from tests.test_function_lowering_v2_cosim import _traces
from tests.test_function_lowering_atan2_cosim import _rust_trace

require_engine()


@pytest.mark.parametrize(("width", "fraction"), [(8, 4), (16, 8), (32, 16)])
@pytest.mark.parametrize("function", ["round", "heaviside"])
@pytest.mark.parametrize("current", [-1.5, 0.0, 2.5])
def test_discrete_equation_matches_python_and_both_integer_kernels(
    width: int, fraction: int, function: str, current: float, tmp_path: Path
) -> None:
    """A dynamic second state exposes Heaviside operand order across C, Rust and RTL."""
    expression = "round(I)" if function == "round" else "heaviside(I, z)"
    neuron = EquationNeuron(
        {"v": expression, "z": "-z"}, state={"v": 0.0, "z": 0.25}, method="map", dt=1.0
    )
    rtl, c = _traces(neuron, current, width, fraction, 8)
    rust = _rust_trace(neuron, current, width, fraction, 8, tmp_path)
    assert len(rtl) == 8 and rtl == c == rust
    for row in rtl:
        neuron.step(current)
        assert int(row[1]) == round(neuron.state["v"] * (1 << fraction))


@pytest.mark.parametrize(
    ("width", "fraction", "method"),
    [
        (8, 4, "map"),
        (16, 8, "map"),
        (32, 16, "map"),
        (16, 8, "euler"),
        (16, 8, "rk4"),
        (16, 8, "gauss_seidel"),
        (16, 8, "exp_euler"),
    ],
)
@pytest.mark.parametrize("function", ["round", "heaviside"])
def test_public_native_discrete_roundtrip_runs_clocked_reference(
    width: int, fraction: int, function: str, method: str, tmp_path: Path
) -> None:
    """Public native SCIR/SV/CIRCT retains reset, changing inputs and a dynamic zero value."""
    expression = "round(I)" if function == "round" else "heaviside(I, z)"
    drive = expression if method == "map" else "-v + " + expression
    neuron = EquationNeuron(
        {"v": drive, "z": "-z"}, state={"v": 0.0, "z": 0.25}, method=method, dt=1.0
    )
    compiled = compile_to_ir(neuron, "discrete_native", width, fraction, function_lowering="v2")
    replayed = recompile_ir(compiled.graph.to_text())
    assert replayed.graph.to_text() == compiled.graph.to_text()
    assert replayed.clocks_per_timestep == 1
    legacy = compile_to_verilog(neuron, "discrete_legacy", width, fraction, function_lowering="v2")
    outputs = dict(compiled.output_widths)
    declarations = " ".join(
        f"wire [{bits - 1}:0] n_{name},l_{name};" for name, bits in outputs.items()
    )
    connections = {
        p: ",".join(
            [
                ".clk(clk)",
                ".rst_n(rst_n)",
                ".I_t(I_t)",
                *(f".{name}({p}_{name})" for name in outputs),
            ]
        )
        for p in ("n", "l")
    }
    comparisons = " || ".join(f"n_{name} !== l_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    steps = []
    for value in [-3.5, -2.5, -1.5, -0.5, 0.0, 0.0, 0.5, 1.5, 2.5, 3.5]:
        code = round(value * (1 << fraction)) & ((1 << width) - 1)
        steps.append(
            f'I_t={width}\'h{code:x}; #2; clk=1; #1; if ({comparisons}) $fatal(1,"discrete mismatch"); {display} clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg [{width - 1}:0] I_t=0; {declarations} discrete_native n({connections['n']}); discrete_legacy l({connections['l']}); initial begin #1;rst_n=0;#1;rst_n=1;\n"
        + "\n".join(steps)
        + "\n$finish;end endmodule\n"
    )
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(compiled.graph.emit_sv() + "\n" + legacy)
    tb.write_text(bench)
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    actual = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=90
    )
    observations = [line for line in actual.stdout.splitlines() if line.startswith("OBS ")]
    assert len(observations) == len(steps)
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"discrete-{function}-{width}-{fraction}-{method}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(compiled.graph.to_text())
        (case / "graph.mlir").write_text(compiled.graph.emit_mlir())
        (case / "tb.sv").write_text(bench + legacy)
        (case / "expected.txt").write_text("\n".join(observations) + "\n")


@pytest.mark.parametrize(
    "expression", ["round(I, 2)", "round()", "heaviside(I)", "heaviside(I,0,1)"]
)
def test_discrete_compile_refuses_wrong_arity(expression: str) -> None:
    """Both public equation compilers refuse operands absent from the model contract."""
    with pytest.raises(ValueError):
        neuron = EquationNeuron({"v": expression}, method="map", dt=1.0)
        compile_to_ir(neuron, function_lowering="v2")
    with pytest.raises(ValueError):
        neuron = EquationNeuron({"v": expression}, method="map", dt=1.0)
        compile_to_verilog(neuron, function_lowering="v2")


@pytest.mark.parametrize("expression", ["round(I)", "heaviside(I,0.25)"])
def test_discrete_units_require_explicit_v2(expression: str) -> None:
    """Adding new units does not change the legacy default or silently substitute tables."""
    neuron = EquationNeuron({"v": expression}, method="map", dt=1.0)
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(neuron)
    with pytest.raises(ValueError):
        compile_to_ir(neuron)


@pytest.mark.parametrize("expression", ["round(v)/tau", "gain*heaviside(v, zero)"])
def test_discrete_strict_source_replay(expression: str) -> None:
    """Quantity display metadata and base-unit expressions survive native reconstruction."""
    from sc_neurocore.neurons._units import UNIT_REGISTRY as u

    model = EquationNeuron(
        {"v": expression},
        state={"v": 1250.0 * u.millivolt},
        parameters={
            "tau": 1.0 * u.second,
            "gain": 1.0 * u.volt / u.second,
            "zero": 0.25 * u.dimensionless,
        },
        dt=0.125 * u.second,
        units="strict",
    )
    compiled = compile_to_ir(model, function_lowering="v2")
    assert recompile_ir(compiled.graph.to_text()).graph.to_text() == compiled.graph.to_text()


@pytest.mark.parametrize("method", ["map", "euler", "rk4", "gauss_seidel", "exp_euler"])
@pytest.mark.parametrize("function", ["round", "heaviside"])
def test_discrete_drift_threshold_reset_and_spike_value_match_kernels(
    method: str, function: str, tmp_path: Path
) -> None:
    """Both new units reach all integration and event-expression positions in real kernels."""
    call = "round(I)" if function == "round" else "heaviside(I,z)"
    model = EquationNeuron(
        {"v": call if method == "map" else "-v+" + call, "z": "-z"},
        state={"v": 0.0, "z": 0.25},
        method=method,
        dt=1.0 if method == "map" else 0.25,
        threshold="heaviside(v-0.25,1)>0",
        reset={"v": "round(-I)/4"},
        spike_value="round(v*4)/4",
    )
    rtl, c, rust = _graded_traces(model, tmp_path)
    assert len(rtl) == 16 and rtl == c == rust
    assert any(int(row[0]) for row in rtl)


def _graded_traces(
    model: EquationNeuron, tmp: Path
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Compile real graded-output drivers, initializing and observing the complete output state."""
    bench = _testbench(model, "m", 1.5, 16, 16, 8)
    bench = bench.replace(
        "    .spike_out(spike_out),", "    .spike_out(spike_out),\n    .spike_value_out(graded),", 1
    )
    bench = bench.replace(
        "reg clk; reg rst_n; wire spike_out;",
        "reg clk; reg rst_n; wire spike_out; wire signed [15:0] graded;",
        1,
    )
    bench = bench.replace('$display("%0d ', '$display("%0d %0d ', 1).replace(
        "$unsigned(spike_out),", "$unsigned(spike_out), $signed(graded),", 1
    )
    top, tb, rtl_bin = tmp / "graded.v", tmp / "tb.v", tmp / "rtl"
    top.write_text(compile_to_verilog(model, "m", function_lowering="v2"))
    tb.write_text(bench)

    def run(command: list[str]) -> str:
        """Execute a bounded real compiler or generated program and retain refusal diagnostics."""
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        return result.stdout

    run(["iverilog", "-g2012", "-o", str(rtl_bin), str(top), str(tb)])
    rtl = _rows(run(["vvp", str(rtl_bin)]))
    c_main = (
        _c_main(model, "m", 384, 16, 16)
        .replace('"%d ', '"%d %d ', 1)
        .replace("spike, ", "spike, (int)st.spike_value_out, ", 1)
    )
    fields = ", ".join(
        [
            *(f"{name}:0, {name}_out:0" for name in model.equations),
            "spike_out:0",
            "spike_value_out:0",
        ]
    )
    outputs = ", ".join(
        ["spike", "st.spike_value_out", *(f"st.{name}_out" for name in model.equations)]
    )
    fmt = " ".join("{}" for _ in range(len(model.equations) + 2))
    rust_main = f'fn main(){{let mut st=MState{{{fields}}};st.reset();for _ in 0..16{{let spike=st.step(384);println!("{fmt}",{outputs});}}}}'
    rows = []
    for language, main in [("c", c_main), ("rust", rust_main)]:
        source = tmp / ("graded.c" if language == "c" else "graded.rs")
        binary = tmp / ("run_" + language)
        source.write_text(
            generate_bittrue_kernel_from_neuron(
                model, "m", language=language, function_lowering="v2"
            )
            + "\n"
            + main
        )
        command = (
            ["gcc", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
            if language == "c"
            else ["rustc", "-O", "-C", "overflow-checks=on"]
        )
        run([*command, str(source), "-o", str(binary)])
        rows.append(_rows(run([str(binary)])))
    return rtl, rows[0], rows[1]
