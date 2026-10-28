# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reciprocal roots across equation compiler boundaries
"""Execute reciprocal-root equations in clocked RTL, fixed kernels and portable C++."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.function_units import evaluate
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import _q_input
from tests.test_function_lowering_atan2_cosim import _run, _rust_trace
from tests.test_function_lowering_v2_cosim import _traces


@pytest.mark.parametrize("width,fraction", [(8, 4), (16, 8), (32, 16), (32, 29)])
@pytest.mark.parametrize("current", [0.0, 0.25, 1.0])
def test_map_matches_all_integer_targets(
    width: int, fraction: int, current: float, tmp_path: Path
) -> None:
    """A complete map preserves rounding, saturation and the explicit zero sentinel."""
    model = EquationNeuron({"v": "rsqrt(I)"}, state={"v": 0.0}, method="map", dt=1.0)
    rtl, c = _traces(model, current, width, fraction, 4)
    rust = _rust_trace(model, current, width, fraction, 4, tmp_path)
    assert len(rtl) == 4 and rtl == c == rust
    expected = evaluate("rsqrt", _q_input(current, width, fraction), width, fraction)
    assert [int(row[1]) for row in rtl] == [expected] * 4


@pytest.mark.parametrize("method", ["euler", "rk4", "gauss_seidel", "exp_euler"])
def test_root_drives_integrators_threshold_and_reset(method: str, tmp_path: Path) -> None:
    """Nonlinear drift and both event-expression positions consume the same root unit."""
    model = EquationNeuron(
        {"v": "-v+rsqrt(v*v+1)+I"},
        state={"v": 0.1},
        dt=0.125,
        method=method,
        threshold="v > rsqrt(4)/2",
        reset={"v": "rsqrt(I*I+1)/8"},
    )
    rtl, c = _traces(model, 0.25, 32, 16, 16)
    rust = _rust_trace(model, 0.25, 32, 16, 16, tmp_path)
    assert len(rtl) == 16 and rtl == c == rust
    assert any(int(row[0]) for row in rtl)


def test_compound_input_narrows_before_reciprocal_root(tmp_path: Path) -> None:
    """A wrapped positive sum uses the declared out-of-domain sentinel in every target."""
    model = EquationNeuron({"v": "rsqrt(I+I)"}, method="map", dt=1.0)
    rtl, c = _traces(model, 100.0, 16, 8, 4)
    rust = _rust_trace(model, 100.0, 16, 8, 4, tmp_path)
    assert rtl == c == rust
    assert [int(row[1]) for row in rtl] == [0] * 4


@pytest.mark.parametrize("nested", [False, True])
def test_portable_cpp_preserves_positive_float_range(nested: bool, tmp_path: Path) -> None:
    """Actual C++ computes subnormal and maximal finite inputs without reciprocal overflow."""
    expression = "rsqrt(rsqrt(x))" if nested else "rsqrt(x)"
    generated, free = emit_c_expr(expression, {"x"}, math_ns="std", fp_type="double")
    assert not free
    inputs = [math.ulp(0.0), sys.float_info.min, 0.25, 1.0, 4.0, sys.float_info.max]
    calls = "\n".join(f'std::printf("%.17g\\n",root({value!r}));' for value in inputs)
    source, binary = tmp_path / "root.cpp", tmp_path / "root"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + f"double root(double x){{return {generated};}}\nint main(){{{calls}}}\n"
    )
    _run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)])
    actual = [float(value) for value in _run([str(binary)]).split()]
    expected = [1 / math.sqrt(value) for value in inputs]
    if nested:
        expected = [1 / math.sqrt(value) for value in expected]
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("expression", ["rsqrt()", "rsqrt(I,1)", "rsqrt(I,x=1)"])
def test_compilers_refuse_wrong_call_signatures(expression: str) -> None:
    """All expression surfaces preserve unary positional-only admission."""
    with pytest.raises(ValueError):
        emit_c_expr(expression, {"I"})
    for language in ("c", "rust"):
        with pytest.raises(ValueError):
            emit_c_fixed_expr(expression, {}, {}, Q88(function_lowering="v2"), lang=language)
    with pytest.raises(ValueError):
        compile_to_verilog(EquationNeuron({"v": expression}, method="map"), function_lowering="v2")


def test_v2_opt_in_and_public_numeric_contract() -> None:
    """The unchanged legacy default refuses the new unit and v2 states its positive domain."""
    model = EquationNeuron({"v": "rsqrt(I)"}, method="map")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_verilog(model)
    for language in ("c", "rust"):
        with pytest.raises(ValueError, match="needs function_lowering='v2'"):
            emit_c_fixed_expr("rsqrt(I)", {}, {}, Q88(), lang=language)
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert len(contract.function_units) == 1
    unit = contract.function_units[0]
    assert unit.function == "rsqrt"
    assert unit.accuracy.declared_error_bound == 1 / 32
    assert unit.accuracy.operand_ranges == ((1 / 16, 127 / 16),)
