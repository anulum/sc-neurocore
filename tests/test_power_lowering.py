# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real power execution across equation compiler targets
"""Execute powers in Python, portable C++, fixed C/Rust and clocked Verilog."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.power_units import power_code
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import _q_input, _rows, _rust_main
from tests.test_function_lowering_atan2_cosim import _run, _rust_trace
from tests.test_function_lowering_v2_cosim import _traces


@pytest.mark.parametrize("width,fraction", [(8, 4), (16, 15), (32, 16), (32, 29)])
@pytest.mark.parametrize("exponent", [-3.0, -1.0, 0.0, 1.0, 1.5, 9.0, 0.5000005])
def test_literal_power_runs_in_all_fixed_targets(
    width: int, fraction: int, exponent: float, tmp_path: Path
) -> None:
    """Literal precision remains independent of the base and output word format."""
    model = EquationNeuron({"v": f"I**{exponent!r}"}, method="map", dt=0.125)
    model.step(0.25)
    assert model.state["v"] == pytest.approx(0.25**exponent)
    rtl, c = _traces(model, 0.25, width, fraction, 3)
    rust = _rust_trace(model, 0.25, width, fraction, 3, tmp_path)
    expected = power_code(_q_input(0.25, width, fraction), width, fraction, exponent=exponent)
    assert len(rtl) == 3 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == [expected] * 3


@pytest.mark.parametrize("width,fraction", [(16, 8), (16, 15), (32, 16), (32, 29)])
def test_dynamic_power_narrows_the_changing_exponent(
    width: int, fraction: int, tmp_path: Path
) -> None:
    """The exponent addition wraps to its signed word before entering every unit."""
    model = EquationNeuron({"v": "I**(w+.75)", "w": "-w"}, state={"w": 0.5}, method="map", dt=0.125)
    rtl, c = _traces(model, 0.25, width, fraction, 6)
    rust = _rust_trace(model, 0.25, width, fraction, 6, tmp_path)
    expected = [
        power_code(
            _q_input(0.25, width, fraction),
            width,
            fraction,
            exponent_code=_q_input(1.25 if index % 2 == 0 else 0.25, width, fraction),
        )
        for index in range(6)
    ]
    assert len(rtl) == 6 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == expected


def test_both_compound_operands_wrap_before_domain_selection(tmp_path: Path) -> None:
    """A positive mathematical sum can wrap to a forbidden negative hardware base."""
    model = EquationNeuron(
        {"v": "(I+I)**(w+w)", "w": "-w"}, state={"w": 100.0}, method="map", dt=0.125
    )
    rtl, c = _traces(model, 100.0, 16, 8, 4)
    rust = _rust_trace(model, 100.0, 16, 8, 4, tmp_path)
    assert len(rtl) == 4 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == [0] * 4


def test_random_dynamic_exponent_consumes_the_same_draws(tmp_path: Path) -> None:
    """A dynamic power preserves the neuron counter and numbered random call site."""
    model = EquationNeuron(
        {"v": "I**rand()"}, method="map", dt=0.125, rng_seed=47, random_stream=13
    )
    rtl, c = _traces(model, 0.5, 16, 8, 8)
    rust = _rust_trace(model, 0.5, 16, 8, 8, tmp_path, has_rng=True)
    assert len(rtl) == 8 and rtl == c == rust
    assert len({row[1] for row in rtl}) > 1


@pytest.mark.parametrize("shadowed", [False, True])
def test_named_exponent_obeys_lexical_shadowing(shadowed: bool, tmp_path: Path) -> None:
    """Unbound pi keeps independent precision; a declared pi is the actual state word."""
    model = EquationNeuron(
        {"v": "I**pi", **({"pi": "pi"} if shadowed else {})},
        state={"pi": 0.5} if shadowed else {},
        method="map",
        dt=0.125,
    )
    rtl, c = _traces(model, 0.25, 16, 15, 3)
    rust = _rust_trace(model, 0.25, 16, 15, 3, tmp_path)
    expected = power_code(
        _q_input(0.25, 16, 15),
        16,
        15,
        exponent=None if shadowed else math.pi,
        exponent_code=_q_input(0.5, 16, 15) if shadowed else None,
    )
    assert len(rtl) == 3 and rtl == c == rust
    assert [int(row[1]) for row in rtl] == [expected] * 3


@pytest.mark.parametrize("method", ["euler", "rk4", "gauss_seidel", "exp_euler"])
def test_power_in_drift_and_event_paths(method: str, tmp_path: Path) -> None:
    """General powers participate in integration, differentiation and real spike resets."""
    model = EquationNeuron(
        {"v": "-v+.2*(v*v+1)**1.5+I"},
        state={"v": 0.125},
        method=method,
        dt=0.125,
        threshold="(v*v+1)**1.5>1.2",
        reset={"v": "(I*I+1)**(I-.5)/8"},
        spike_value="(v*v+1)**1.5",
    )
    rtl, c = _traces(model, 0.5, 32, 16, 12)
    source = generate_bittrue_kernel_from_neuron(
        model, "m", data_width=32, fraction=16, language="rust", function_lowering="v2"
    )
    main = _rust_main(model, "m", _q_input(0.5, 32, 16), 12, 32)
    main = main.replace("spike_out: 0", "spike_out: 0, spike_value_out: 0", 1)
    path, binary = tmp_path / "power.rs", tmp_path / "power"
    path.write_text(source + main)
    _run(["rustc", "-C", "overflow-checks=on", "-O", str(path), "-o", str(binary)])
    rust = _rows(_run([str(binary)]))
    assert len(rtl) == 12 and rtl == c == rust
    assert sum(int(row[0]) for row in rtl) > 0


def test_portable_cpp_runs_the_original_real_exponent(tmp_path: Path) -> None:
    """Compiled C++ uses the standard library for general powers and exact near-root literals."""
    expressions = ["x**-3", "x**0", "x**1.5", "x**9", "x**0.5000005", "x**pi", "x**y"]
    pairs = [(0.25, -1.25), (0.75, 0.5), (1.25, 3.25)]
    functions, calls, expected = [], [], []
    for index, expression in enumerate(expressions):
        code, free = emit_c_expr(expression, {"x", "y"}, math_ns="std", fp_type="double")
        assert not free
        functions.append(f"double f{index}(double x,double y){{(void)y;return {code};}}")
        for x, y in pairs:
            calls.append(f'std::printf("%.17g\\n",f{index}({x!r},{y!r}));')
            exponent = y if index == 6 else [-3, 0, 1.5, 9, 0.5000005, math.pi][index]
            expected.append(math.pow(x, exponent))
    source, binary = tmp_path / "power.cpp", tmp_path / "power"
    source.write_text(
        "#include <cmath>\n#include <cstdio>\n"
        + "\n".join(functions)
        + "\nint main(){"
        + "".join(calls)
        + "}\n"
    )
    _run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)])
    assert [float(item) for item in _run([str(binary)]).split()] == pytest.approx(
        expected, rel=2e-15, abs=1e-15
    )


def test_direct_fixed_expression_preserves_large_integer_parity(tmp_path: Path) -> None:
    """The expression API preserves integer parity beyond binary64 precision.

    Whole-neuron admission retains its separate exponent resource limit.
    """
    foundation = generate_bittrue_kernel_from_neuron(
        EquationNeuron({"v": "I"}, method="map"), "base", function_lowering="v2"
    ).split("typedef struct", 1)[0]
    odd = (1 << 100) + 1
    functions = []
    for index, exponent in enumerate([odd, -odd, odd + 1]):
        expression, statements, tables, free, _, _ = emit_c_fixed_expr(
            f"I**{exponent}", {}, {}, Q88(function_lowering="v2")
        )
        assert not tables and not free
        functions.append(
            f"int64_t f{index}(int64_t I_t){{" + "\n".join(statements) + f"return {expression};}}"
        )
    source, binary = tmp_path / "literal.c", tmp_path / "literal"
    source.write_text(
        foundation
        + "#include <stdio.h>\n"
        + "\n".join(functions)
        + '\nint main(void){printf("%lld %lld %lld\\n",(long long)f0(-256),'
        + "(long long)f1(-256),(long long)f2(-256));}\n"
    )
    _run(
        [
            "gcc",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(source),
            "-o",
            str(binary),
        ]
    )
    assert [int(item) for item in _run([str(binary)]).split()] == [-256, -256, 256]


@pytest.mark.parametrize("expression", ["I**-3", "I**1.5", "I**9", "I**I"])
def test_legacy_default_keeps_general_power_refusals(expression: str) -> None:
    """General fixed-point powers require the explicit v2 opt-in on each public surface."""
    with pytest.raises(ValueError, match="Only integer powers"):
        compile_to_verilog(EquationNeuron({"v": expression}, method="map"))
    for language in ("c", "rust"):
        with pytest.raises(ValueError, match="Only integer powers"):
            emit_c_fixed_expr(expression, {}, {}, Q88(), lang=language)
