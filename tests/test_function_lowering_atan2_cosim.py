# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — atan2 across real equation compiler boundaries

"""Exercise Python equations, portable C++, fixed C/Rust and clocked Verilog."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import _q_input, _rows, _rust_main
from tests.test_function_lowering_v2_cosim import _traces


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout


def _rust_trace(
    neuron: EquationNeuron,
    current: float,
    dw: int,
    fr: int,
    steps: int,
    tmp: Path,
    *,
    has_rng: bool = False,
) -> list[list[str]]:
    source = generate_bittrue_kernel_from_neuron(
        neuron, "m", data_width=dw, fraction=fr, language="rust", function_lowering="v2"
    )
    path = tmp / "neuron.rs"
    main = _rust_main(neuron, "m", _q_input(current, dw, fr), steps, dw)
    if has_rng:
        main = main.replace("spike_out: 0", "spike_out: 0, rng_step: 0", 1)
    path.write_text(source + "\n" + main)
    executable = tmp / "neuron"
    _run(["rustc", "-C", "overflow-checks=on", "-O", str(path), "-o", str(executable)])
    return _rows(_run([str(executable)]))


@pytest.mark.parametrize("dw,fr", [(16, 8), (16, 12), (32, 16), (32, 24)])
@pytest.mark.parametrize("current", [-2.0, 0.0, 2.0])
def test_map_with_changing_second_operand_matches_all_targets(
    dw: int,
    fr: int,
    current: float,
    tmp_path: Path,
) -> None:
    """A real simultaneous map alternates the second operand through both half-planes."""
    neuron = EquationNeuron(
        equations={"v": "atan2(I, x)", "x": "-x"}, state={"v": 0.0, "x": 2.0}, method="map", dt=1.0
    )
    rtl, c = _traces(neuron, current, dw, fr, 8)
    rust = _rust_trace(neuron, current, dw, fr, 8, tmp_path)
    expected = [
        fu.evaluate(
            "atan2",
            _q_input(current, dw, fr),
            dw,
            fr,
            additional_inputs=(_q_input(2.0 if k % 2 == 0 else -2.0, dw, fr),),
        )
        for k in range(8)
    ]
    assert len(rtl) == 8 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == expected


def test_each_wide_argument_wraps_before_the_unit(tmp_path: Path) -> None:
    """The RTL and both fixed kernels narrow both compound arguments identically."""
    neuron = EquationNeuron(
        equations={"v": "atan2(I + I, x + x)", "x": "-x"},
        state={"v": 0.0, "x": 100.0},
        method="map",
        dt=1.0,
    )
    rtl, c = _traces(neuron, 100.0, 16, 8, 4)
    rust = _rust_trace(neuron, 100.0, 16, 8, 4, tmp_path)
    assert rtl == c == rust
    assert int(rtl[0][1]) == fu.evaluate(
        "atan2", _q_input(200.0, 16, 8), 16, 8, additional_inputs=(_q_input(200.0, 16, 8),)
    )


@pytest.mark.parametrize("method", ["euler", "exp_euler", "rk4"])
def test_angle_in_drift_threshold_and_reset_matches_all_targets(
    method: str, tmp_path: Path
) -> None:
    """Integration and event paths consume the angle without bypassing neuron semantics."""
    neuron = EquationNeuron(
        equations={"v": "0.2*atan2(v, 1) - v + I"},
        state={"v": -0.1},
        dt=0.125,
        method=method,
        threshold="atan2(v, 1) > 0.2",
        reset={"v": "atan2(-I, 1)"},
    )
    rtl, c = _traces(neuron, 0.75, 32, 16, 40)
    rust = _rust_trace(neuron, 0.75, 32, 16, 40, tmp_path)
    assert len(rtl) == 40 and rtl == c == rust
    assert sum(int(row[0]) for row in rtl) > 0


@pytest.mark.parametrize("nested", [False, True])
def test_portable_cpp_expression_matches_the_float_reference(tmp_path: Path, nested: bool) -> None:
    """The C-expression emitter passes both coordinates to an actual math library."""
    expression = "atan2(sin(y), cos(x))" if nested else "atan2(y, x)"
    expr, free = emit_c_expr(expression, {"y", "x"}, math_ns="std", fp_type="double")
    assert not free
    pairs = [
        (0.0, 0.0),
        (0.0, -2.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 2.0),
        (1.0, -2.0),
        (-1.0, -2.0),
        (-1e-20, 1e20),
        (-0.0, -2.0),
        (0.0, -0.0),
        (-0.0, -0.0),
    ]
    calls = "\n".join(f'std::printf("%.17g\\n", f({y!r},{x!r}));' for y, x in pairs)
    source = tmp_path / "angle.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + f"double f(double y,double x){{return {expr};}}\nint main(){{{calls}}}\n"
    )
    executable = tmp_path / "angle"
    _run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)])
    result = [float(value) for value in _run([str(executable)]).split()]
    coordinates = [(math.sin(y), math.cos(x)) for y, x in pairs] if nested else pairs
    assert result == pytest.approx(
        [math.atan2(0.0 if y == 0 else y, 0.0 if x == 0 else x) for y, x in coordinates], abs=1e-15
    )


@pytest.mark.parametrize("expression", ["atan2(I)", "atan2(I, 1, 2)", "atan2(I, x=1)"])
def test_all_expression_backends_refuse_unsupported_arity(expression: str) -> None:
    """No backend may drop a coordinate or an output/keyword argument."""
    with pytest.raises(ValueError):
        emit_c_expr(expression, set())
    for lang in ("c", "rust"):
        with pytest.raises(ValueError):
            emit_c_fixed_expr(expression, {}, {}, Q88(function_lowering="v2"), lang=lang)
    if "x=" not in expression and expression.count(",") == 0:
        neuron = EquationNeuron(equations={"v": expression}, state={"v": 0.0}, method="map")
        with pytest.raises(ValueError, match="exactly 2"):
            compile_to_verilog(neuron, "m", function_lowering="v2")
    else:
        with pytest.raises(ValueError):
            EquationNeuron(equations={"v": expression}, state={"v": 0.0}, method="map")


def test_old_lowering_refuses_the_new_function() -> None:
    """Opt-in v2 support does not silently change historical default RTL."""
    neuron = EquationNeuron(equations={"v": "atan2(I, 1)"}, state={"v": 0.0}, method="map")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(neuron, "m")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        generate_bittrue_kernel_from_neuron(neuron, "m")
