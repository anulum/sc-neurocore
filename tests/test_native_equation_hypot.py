# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Euclidean magnitude equation compilation
"""Execute complete norm models through native IR, portable C++ and fixed kernels."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.function_units import evaluate
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine
from tests.test_function_lowering_atan2_cosim import _rust_trace
from tests.test_function_lowering_v2_cosim import _traces
from tests.test_native_equation_discrete_functions import _graded_traces

require_engine()


@pytest.mark.parametrize("width,fraction", [(8, 4), (16, 8), (32, 16), (32, 24)])
@pytest.mark.parametrize("current", [-0.75, 0.0, 0.75])
def test_norm_map_uses_both_dynamic_coordinates_in_all_fixed_targets(
    width: int, fraction: int, current: float, tmp_path: Path
) -> None:
    """Changing the second coordinate's magnitude reaches C, Rust and clocked Verilog."""
    model = EquationNeuron(
        {"v": "hypot(I,z)", "z": "z/2"}, state={"v": 0.0, "z": 0.5}, method="map", dt=1.0
    )
    rtl, c = _traces(model, current, width, fraction, 8)
    rust = _rust_trace(model, current, width, fraction, 8, tmp_path)
    assert len(rtl) == 8 and rtl == c == rust
    second = 1 << (fraction - 1)
    expected = []
    for _ in range(8):
        expected.append(
            evaluate(
                "hypot",
                round(current * (1 << fraction)),
                width,
                fraction,
                additional_inputs=(second,),
            )
        )
        second //= 2
    assert [int(row[1]) for row in rtl] == expected


@pytest.mark.parametrize("method", ["map", "euler", "rk4", "gauss_seidel", "exp_euler"])
def test_norm_drives_all_integrators_events_resets_and_graded_outputs(
    method: str, tmp_path: Path
) -> None:
    """Every maintained integration path consumes the norm in all event-expression positions."""
    call = "hypot(v,1)"
    model = EquationNeuron(
        {"v": call if method == "map" else "-v+" + call},
        state={"v": 0.0},
        method=method,
        dt=1.0 if method == "map" else 0.25,
        threshold="hypot(v,0)>0.5",
        reset={"v": "hypot(I,1)/8"},
        spike_value="hypot(v,1)",
    )
    rtl, c, rust = _graded_traces(model, tmp_path)
    assert len(rtl) == 16 and rtl == c == rust
    assert any(int(row[0]) for row in rtl)


@pytest.mark.parametrize(
    "width,fraction,method",
    [
        (8, 4, "map"),
        (32, 16, "map"),
        (32, 24, "map"),
        (16, 8, "euler"),
        (16, 8, "rk4"),
        (16, 8, "gauss_seidel"),
        (16, 8, "exp_euler"),
    ],
)
def test_native_norm_replays_and_runs_against_the_clocked_compiler(
    width: int, fraction: int, method: str, tmp_path: Path
) -> None:
    """Source-bound native graphs and direct RTL agree under changing drive and real reset."""
    drive = "hypot(v,1)" if method == "map" else "-v+hypot(v,1)+I"
    model = EquationNeuron(
        {"v": drive},
        state={"v": -0.25},
        method=method,
        dt=1.0 if method == "map" else 0.125,
        threshold="hypot(v,0)>.75",
        reset={"v": "hypot(I,1)/8"},
        spike_value="hypot(v,1)",
    )
    native = compile_to_ir(model, "norm_native", width, fraction, function_lowering="v2")
    assert recompile_ir(native.graph.to_text()).graph.to_text() == native.graph.to_text()
    legacy = compile_to_verilog(model, "norm_legacy", width, fraction, function_lowering="v2")
    outputs = dict(native.output_widths)
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
    compare = " || ".join(f"n_{name} !== l_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    steps = []
    for i, current in enumerate([-0.75, 0.0, 0.75, 0.5, -0.5, 0.0, 0.75, -0.75]):
        code = round(current * (1 << fraction)) & ((1 << width) - 1)
        reset = "rst_n=0;#1;rst_n=1;#1;" if i == 4 else ""
        steps.append(
            f'{reset} I_t={width}\'h{code:x};#2;clk=1;#1;if ({compare}) $fatal(1,"norm mismatch");{display}clk=0;#1;'
        )
    bench = (
        f"module tb;reg clk=0,rst_n=1;reg [{width - 1}:0] I_t=0;{declarations}"
        f"norm_native n({connections['n']});norm_legacy l({connections['l']});"
        "initial begin #1;rst_n=0;#1;rst_n=1;\n" + "\n".join(steps) + "\n$finish;end endmodule\n"
    )
    top, tb, binary = tmp_path / "norm.sv", tmp_path / "tb.sv", tmp_path / "run"
    top.write_text(native.graph.emit_sv() + "\n" + legacy)
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
    if receipt_dir := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(receipt_dir) / f"hypot-{width}-{fraction}-{method}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(native.graph.to_text())
        (case / "graph.mlir").write_text(native.graph.emit_mlir())
        (case / "tb.sv").write_text(bench + legacy)
        (case / "expected.txt").write_text("\n".join(observations) + "\n")


def test_portable_cpp_norm_survives_large_and_tiny_floating_coordinates(tmp_path: Path) -> None:
    """The C++ path uses a real norm implementation, preserving float exponent range."""
    expression, free = emit_c_expr("hypot(a,b)", {"a", "b"}, math_ns="std", fp_type="double")
    assert not free
    pairs = [(3.0, 4.0), (-3.0, 4.0), (0.0, 0.0), (3e200, 4e200), (3e-200, 4e-200)]
    calls = "\n".join(f'std::printf("%.17g\\n",norm({a!r},{b!r}));' for a, b in pairs)
    source, binary = tmp_path / "norm.cpp", tmp_path / "norm"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + f"double norm(double a,double b){{return {expression};}}\nint main(){{{calls}}}\n"
    )
    subprocess.run(
        ["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=90)
    assert [float(x) for x in result.stdout.split()] == pytest.approx(
        [math.hypot(a, b) for a, b in pairs], rel=1e-15, abs=0
    )


@pytest.mark.parametrize(
    "expression", ["hypot(I)", "hypot()", "hypot(I,0,1)", "hypot(I,0,out)", "hypot(I,0,None)"]
)
def test_norm_compilers_refuse_wrong_arity_and_output_arrays(expression: str) -> None:
    """Only two coordinate inputs enter compiled arithmetic; array-write arguments refuse."""
    with pytest.raises(ValueError):
        model = EquationNeuron({"v": expression}, method="map")
        compile_to_ir(model, function_lowering="v2")


def test_norm_requires_opt_in_and_reports_its_numeric_contract() -> None:
    """Adding a function preserves legacy output defaults and exposes its measured accuracy."""
    model = EquationNeuron({"v": "hypot(I,1)"}, method="map")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(model)
    with pytest.raises(ValueError):
        compile_to_ir(model)
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert any(unit.function == "hypot" for unit in contract.function_units)


def test_strict_norm_source_metadata_survives_native_replay() -> None:
    """The common physical dimension and original display quantities survive source replay."""
    from sc_neurocore.neurons._units import UNIT_REGISTRY as u

    model = EquationNeuron(
        {"v": "hypot(v,offset)/tau"},
        state={"v": -3000 * u.millivolt},
        parameters={"offset": 4 * u.volt, "tau": 1 * u.second},
        dt=0.125 * u.second,
        units="strict",
    )
    compiled = compile_to_ir(model, function_lowering="v2")
    assert recompile_ir(compiled.graph.to_text()).graph.to_text() == compiled.graph.to_text()


@pytest.mark.parametrize("second", ["0", "0.000001", "I"])
def test_native_norm_denominator_refuses_a_possible_encoded_zero(second: str) -> None:
    """A real nonzero value below the quantizer's resolution supplies no division proof."""
    model = EquationNeuron({"v": f"1/hypot(v,{second})"}, method="map")
    with pytest.raises(ValueError, match="may encode zero"):
        compile_to_ir(model, function_lowering="v2")


@pytest.mark.parametrize("expression", ["1/hypot(v,-1)", "1/hypot(offset,v)"])
def test_native_norm_denominator_accepts_an_encoded_nonzero_coordinate(expression: str) -> None:
    """Either a signed constant or a fixed parameter can establish the norm's lower bound."""
    model = EquationNeuron({"v": expression}, parameters={"offset": 1.0}, method="map")
    compiled = compile_to_ir(model, function_lowering="v2")
    assert recompile_ir(compiled.graph.to_text()).graph.to_text() == compiled.graph.to_text()
