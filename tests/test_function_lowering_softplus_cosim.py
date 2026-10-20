# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — softplus through whole equation compiler boundaries

"""Run stable softplus equations through RTL, checked kernels and portable C++."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import function_units as fu
from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationTypeFailure
from tests.bit_true_cosim_support import _q_input
from tests.test_function_lowering_atan2_cosim import _rust_trace, _run
from tests.test_function_lowering_v2_cosim import _traces


@pytest.mark.parametrize("dw,fr", [(16, 8), (16, 12), (32, 16), (32, 24)])
@pytest.mark.parametrize("current", [-2.0, 0.0, 2.0])
def test_map_output_matches_all_targets(dw: int, fr: int, current: float, tmp_path: Path) -> None:
    """The real model-to-native path returns the same bounded integer reference value."""
    neuron = EquationNeuron({"v": "softplus(I)"}, state={"v": 0.0}, method="map", dt=1.0)
    rtl, c = _traces(neuron, current, dw, fr, 4)
    rust = _rust_trace(neuron, current, dw, fr, 4, tmp_path)
    assert len(rtl) == 4 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == [
        fu.evaluate("softplus", _q_input(current, dw, fr), dw, fr)
    ] * 4


@pytest.mark.parametrize("method", ["euler", "exp_euler", "rk4", "gauss_seidel"])
def test_nonlinear_drift_threshold_and_reset_match(method: str, tmp_path: Path) -> None:
    """Softplus reaches every integration stage and both event-expression positions."""
    neuron = EquationNeuron(
        {"v": "0.25*softplus(v) - v + I"},
        state={"v": -0.1},
        dt=0.125,
        method=method,
        threshold="softplus(v) > 0.8",
        reset={"v": "-softplus(-I)"},
    )
    rtl, c = _traces(neuron, 0.75, 32, 16, 40)
    rust = _rust_trace(neuron, 0.75, 32, 16, 40, tmp_path)
    assert len(rtl) == 40 and rtl == c == rust
    assert sum(int(row[0]) for row in rtl) > 0


def test_compound_argument_wraps_before_softplus(tmp_path: Path) -> None:
    """A wide sum follows the same word narrowing in RTL and both integer kernels."""
    neuron = EquationNeuron({"v": "softplus(I+I)"}, state={"v": 0.0}, method="map", dt=1.0)
    rtl, c = _traces(neuron, 100.0, 16, 8, 4)
    rust = _rust_trace(neuron, 100.0, 16, 8, 4, tmp_path)
    assert rtl == c == rust
    assert int(rtl[0][1]) == fu.evaluate("softplus", _q_input(200.0, 16, 8), 16, 8) == 0


def test_extreme_state_has_a_stable_exponential_euler_jacobian(tmp_path: Path) -> None:
    """The public exponential integrator never constructs exp(1000) for the Jacobian."""
    neuron = EquationNeuron(
        {"v": "softplus(v)-v+I"}, state={"v": 1000.0}, method="exp_euler", dt=0.125
    )
    rtl, c = _traces(neuron, 0.125, 32, 16, 8)
    rust = _rust_trace(neuron, 0.125, 32, 16, 8, tmp_path)
    assert rtl == c == rust
    for row in rtl:
        neuron.step(0.125)
        assert int(row[1]) == round(neuron.state["v"] * 65536)


@pytest.mark.parametrize("nested", [False, True])
def test_portable_cpp_avoids_exponential_overflow(nested: bool, tmp_path: Path) -> None:
    """An actual C++ math library matches a separate stable NumPy reference on finite tails."""
    expression = "softplus(softplus(x))" if nested else "softplus(x)"
    expr, free = emit_c_expr(expression, {"x"}, math_ns="std", fp_type="double")
    assert not free
    values = [
        -sys.float_info.max,
        -1000.0,
        -30.0,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        2.0,
        30.0,
        1000.0,
        sys.float_info.max,
    ]
    calls = "\n".join(f'std::printf("%.17g\\n", f({x!r}));' for x in values)
    source = tmp_path / "softplus.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + f"double f(double x){{return {expr};}}\nint main(){{{calls}}}\n"
    )
    executable = tmp_path / "softplus"
    _run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)])
    actual = [float(x) for x in _run([str(executable)]).split()]
    with np.errstate(under="ignore"):
        expected = [float(np.logaddexp(0.0, x)) for x in values]
        if nested:
            expected = [float(np.logaddexp(0.0, x)) for x in expected]
    assert all(math.isfinite(x) for x in actual)
    assert actual == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_nested_cpp_growth_is_bounded_and_the_program_executes(tmp_path: Path) -> None:
    """A compact nested input stays compact and computes the independent log(9) identity."""
    expression = "x"
    for _ in range(8):
        expression = f"softplus({expression})"
    generated, free = emit_c_expr(expression, {"x"}, math_ns="std", fp_type="double")
    assert not free and len(generated) < 8192
    source = tmp_path / "nested.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + f"double f(double x){{return {generated};}}\n"
        + 'int main(){std::printf("%.17g\\n",f(0.0));}\n'
    )
    executable = tmp_path / "nested"
    _run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)])
    assert float(_run([str(executable)])) == pytest.approx(math.log(9.0), abs=2e-15)


@pytest.mark.parametrize("expression", ["softplus()", "softplus(I, None)", "softplus(I, x=1)"])
def test_emitters_refuse_unsupported_arity(expression: str) -> None:
    """Every expression backend refuses an argument that its unary contract cannot lower."""
    with pytest.raises(ValueError):
        emit_c_expr(expression, set())
    for lang in ("c", "rust"):
        with pytest.raises(ValueError):
            emit_c_fixed_expr(expression, {}, {}, Q88(function_lowering="v2"), lang=lang)
    if "=" not in expression:
        neuron = EquationNeuron({"v": expression}, state={"v": 0.0}, method="map", dt=1.0)
        with pytest.raises(ValueError):
            compile_to_verilog(neuron, "m", function_lowering="v2")
    else:
        with pytest.raises(ValueError):
            EquationNeuron({"v": expression}, state={"v": 0.0}, method="map", dt=1.0)


def test_numeric_binding_cannot_become_a_function() -> None:
    """The runtime and compiler retain actual numeric parameter shadowing."""
    neuron = EquationNeuron(
        {"v": "softplus(I)"}, parameters={"softplus": 1.0}, state={"v": 0.0}, method="map", dt=1.0
    )
    with pytest.raises(EquationTypeFailure):
        neuron.step(0.25)
    with pytest.raises(ValueError, match="shadowed"):
        emit_c_expr("softplus(I)", set(), param_map={"softplus": "1.0"})
    with pytest.raises(ValueError, match="shadowed"):
        compile_to_verilog(neuron, "m", function_lowering="v2")
    for lang in ("c", "rust"):
        with pytest.raises(ValueError, match="shadowed"):
            generate_bittrue_kernel_from_neuron(neuron, "m", language=lang, function_lowering="v2")


def test_output_arrays_are_refused_and_old_lowering_is_preserved() -> None:
    """No array-write argument or implicit change to lut-v1 is admitted."""
    with pytest.raises(ValueError, match="output argument"):
        EquationNeuron({"v": "softplus(I, [0.0])"}, state={"v": 0.0})
    neuron = EquationNeuron({"v": "softplus(I)"}, state={"v": 0.0}, method="map", dt=1.0)
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(neuron, "m")
