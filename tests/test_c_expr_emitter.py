# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the C/C++ expression emitter

"""Tests for the C++ (ap_fixed) expression emitter."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.c_expr_emitter import CExprEmitter, emit_c_expr


def _emit(expr: str, state_vars: set[str] | None = None, **kw: object) -> str:
    code, _ = emit_c_expr(expr, state_vars or set(), **kw)  # type: ignore[arg-type]
    return code


class TestArithmetic:
    def test_add_sub_mul_div(self) -> None:
        assert _emit("a + b") == "(a + b)"
        assert _emit("a - b") == "(a - b)"
        assert _emit("a * b") == "(a * b)"
        assert _emit("a / b") == "(a / b)"

    def test_unary_minus(self) -> None:
        assert _emit("-a") == "(-a)"

    def test_unary_plus_passthrough(self) -> None:
        assert _emit("+a") == "a"

    def test_constant_cast_to_fp_type(self) -> None:
        assert _emit("1.5") == "fp_t(1.5)"

    def test_nested(self) -> None:
        assert _emit("(a + b) * c") == "((a + b) * c)"


class TestPow:
    def test_square_is_repeated_multiply(self) -> None:
        assert _emit("v ** 2", {"v"}) == "(v * v)"

    def test_cube(self) -> None:
        assert _emit("v ** 3", {"v"}) == "(v * v * v)"

    def test_eighth_power(self) -> None:
        assert _emit("v ** 8", {"v"}).count("v *") == 7

    def test_sqrt_from_half_power(self) -> None:
        assert _emit("v ** 0.5", {"v"}) == "hls::sqrt(v)"

    def test_cbrt_from_third_power(self) -> None:
        assert _emit("v ** (1.0 / 3.0)", {"v"}) == "hls::cbrt(v)"

    def test_unsupported_power_raises(self) -> None:
        with pytest.raises(ValueError, match="Only integer powers"):
            _emit("v ** 9", {"v"})


class TestNames:
    def test_state_var_verbatim(self) -> None:
        assert _emit("v", {"v"}) == "v"

    def test_input_current_maps_to_I_t(self) -> None:
        assert _emit("I") == "I_t"

    def test_param_map(self) -> None:
        assert _emit("tau", set(), param_map={"tau": "P_tau"}) == "P_tau"

    def test_free_vars_recorded_in_order(self) -> None:
        code, free = emit_c_expr("a + b - a", set())
        assert code == "((a + b) - a)"
        assert free == ["a", "b"]


class TestComparisons:
    def test_all_comparisons(self) -> None:
        assert _emit("v > 1.0", {"v"}) == "(v > fp_t(1.0))"
        assert _emit("v >= 1.0", {"v"}) == "(v >= fp_t(1.0))"
        assert _emit("v < 1.0", {"v"}) == "(v < fp_t(1.0))"
        assert _emit("v <= 1.0", {"v"}) == "(v <= fp_t(1.0))"

    def test_unsupported_comparison_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported comparison"):
            _emit("v == 1.0", {"v"})


class TestFunctionCalls:
    def test_direct_math_functions(self) -> None:
        assert _emit("exp(v)", {"v"}) == "hls::exp(v)"
        assert _emit("tanh(v)", {"v"}) == "hls::tanh(v)"
        assert _emit("cosh(v)", {"v"}) == "hls::cosh(v)"
        assert _emit("sin(v)", {"v"}) == "hls::sin(v)"
        assert _emit("log(v)", {"v"}) == "hls::log(v)"

    def test_std_namespace_option(self) -> None:
        assert _emit("exp(v)", {"v"}, math_ns="std") == "std::exp(v)"

    def test_sigmoid_and_expit_use_helper(self) -> None:
        assert _emit("sigmoid(v)", {"v"}) == "sc_sigmoid(v)"
        assert _emit("expit(v)", {"v"}) == "sc_sigmoid(v)"

    def test_exprel_uses_helper(self) -> None:
        assert _emit("exprel(v)", {"v"}) == "sc_exprel(v)"

    def test_abs(self) -> None:
        assert _emit("abs(v)", {"v"}) == "hls::abs(v)"

    def test_clip_three_args(self) -> None:
        out = _emit("clip(v, 0.0, 1.0)", {"v"})
        assert "? fp_t(0.0)" in out and "? fp_t(1.0)" in out

    def test_max_min(self) -> None:
        assert _emit("max(a, b)") == "((a > b) ? a : b)"
        assert _emit("min(a, b)") == "((a < b) ? a : b)"

    def test_clip_single_arg_passthrough(self) -> None:
        # clip with only the value (no bounds) passes the argument through.
        assert _emit("clip(v)", {"v"}) == "v"

    def test_max_single_arg_passthrough(self) -> None:
        assert _emit("max(v)", {"v"}) == "v"

    def test_unsupported_function_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported function"):
            _emit("gamma(v)", {"v"})

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _emit("exp()")

    def test_non_name_call_raises(self) -> None:
        with pytest.raises(ValueError, match="Only named function calls"):
            _emit("obj.method(v)", {"v"})


class TestUnsupportedNodes:
    def test_list_node_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported AST node"):
            _emit("[1, 2, 3]")

    def test_unsupported_binop_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported binary op"):
            _emit("a % b")


class TestEmitterState:
    def test_free_vars_attribute(self) -> None:
        e = CExprEmitter({"v"})
        e.visit(__import__("ast").parse("v + leak", mode="eval").body)
        assert e.free_vars == ["leak"]
