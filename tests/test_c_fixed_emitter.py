# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Unit tests for the bit-exact integer C/Rust expression emitter.

These assert on the *shape* of the emitted source (operators, helper calls, LUT
statements, free-variable capture, language differences); the numeric proof that
the emitted C reproduces the Verilog RTL bit-for-bit lives in
``tests/test_bit_true_cosim.py`` (iverilog co-simulation).
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.c_fixed_emitter import (
    _CFixedExprEmitter,
    emit_c_fixed_expr,
    signed_q,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88

Q = Q88(data_width=16, fraction=8)


def _c(expr, state=None, params=None, **kw):
    return emit_c_fixed_expr(expr, state or {}, params or {}, Q, **kw)


class TestSignedQ:
    def test_positive_within_range(self):
        assert signed_q(Q, 1.0) == 256

    def test_negative_two_complement(self):
        # -65.0 in Q8.8 → -16640, reinterpreted signed
        assert signed_q(Q, -65.0) == -16640

    def test_wraps_when_out_of_range(self):
        # +200 exceeds Q8.8 max (127.996); the pattern wraps to a negative value
        assert signed_q(Q, 200.0) < 0


class TestArithmetic:
    def test_add_sub(self):
        expr, *_ = _c("a + b - 1.0", state={"a": "s->a", "b": "s->b"})
        assert "+" in expr and "-" in expr

    def test_multiply_uses_fxmul(self):
        expr, *_ = _c("a * b", state={"a": "s->a", "b": "s->b"})
        assert "fxmul(" in expr

    def test_div_by_constant_becomes_reciprocal_multiply(self):
        # 1/4 in Q8.8 → 64; division by a literal is a reciprocal fxmul, no sc_wrap
        expr, *_ = _c("a / 4.0", state={"a": "s->a"})
        assert "fxmul(" in expr and "64" in expr and "sc_wrap" not in expr

    def test_div_by_variable_uses_shift_divide(self):
        expr, *_ = _c("a / b", state={"a": "s->a", "b": "s->b"})
        assert "sc_wrap(" in expr and "<< 8" in expr and "/" in expr

    def test_unary_negate(self):
        expr, *_ = _c("-a", state={"a": "s->a"})
        assert "(-(" in expr

    def test_unary_plus_is_identity(self):
        expr, *_ = _c("+a", state={"a": "s->a"})
        assert "s->a" in expr

    def test_non_literal_modulo_raises(self):
        with pytest.raises(ValueError, match="Modulo divisor must be a positive numeric literal"):
            _c("a % b", state={"a": "s->a", "b": "s->b"})

    def test_unsupported_unaryop_raises(self):
        with pytest.raises(ValueError, match="Unsupported unary op"):
            _c("~a", state={"a": "s->a"})


class TestPow:
    @pytest.mark.parametrize("exp", [2, 3, 4, 5, 8])
    def test_integer_powers_expand_to_fxmul_chain(self, exp):
        expr, *_ = _c(f"a ** {exp}", state={"a": "s->a"})
        assert expr.count("fxmul(") == exp - 1

    def test_sqrt_power_uses_lut(self):
        expr, stmts, tables, *_ = _c("a ** 0.5", state={"a": "s->a"})
        assert any("sqrt" in t for t in tables)

    def test_cbrt_power_uses_lut(self):
        expr, stmts, tables, *_ = _c("a ** (1.0/3.0)", state={"a": "s->a"})
        assert any("cbrt" in t for t in tables)

    def test_unsupported_power_raises(self):
        with pytest.raises(ValueError, match="Only integer powers"):
            _c("a ** 1.7", state={"a": "s->a"})


class TestNames:
    def test_state_variable(self):
        expr, *_ = _c("v", state={"v": "s->v"})
        assert "s->v" in expr

    def test_parameter(self):
        expr, *_ = _c("tau", params={"tau": 2560})
        assert "2560" in expr

    def test_input_current_sets_flag(self):
        expr, _s, _t, _fv, _lut, used = _c("I")
        assert used is True and "I_t" in expr

    def test_free_variable_recorded(self):
        expr, _s, _t, free, *_ = _c("a + b", state={"a": "s->a"})
        assert free == ["b"]

    def test_free_variable_recorded_once(self):
        _e, _s, _t, free, *_ = _c("b + b")
        assert free == ["b"]


class TestCompare:
    @pytest.mark.parametrize("op,sym", [(">", ">"), (">=", ">="), ("<", "<"), ("<=", "<=")])
    def test_comparisons(self, op, sym):
        expr, *_ = _c(f"v {op} 30.0", state={"v": "s->v"})
        assert sym in expr

    def test_chained_comparison(self):
        expr, *_ = _c("0.0 < v", state={"v": "s->v"})
        assert "<" in expr

    def test_unsupported_comparison_raises(self):
        with pytest.raises(ValueError, match="Unsupported comparison"):
            _c("v == 1.0", state={"v": "s->v"})


class TestCalls:
    @pytest.mark.parametrize(
        "fn", ["exp", "log", "sqrt", "tanh", "cosh", "exprel", "sigmoid", "sin", "cos"]
    )
    def test_transcendental_emits_table(self, fn):
        _e, stmts, tables, *_ = _c(f"{fn}(v)", state={"v": "s->v"})
        assert len(tables) == 1
        assert any("_arg" in s for s in stmts) and any("_idx" in s for s in stmts)

    def test_expit_aliases_sigmoid(self):
        _e, _s, tables, *_ = _c("expit(v)", state={"v": "s->v"})
        assert any("sigmoid" in t for t in tables)

    def test_abs(self):
        expr, *_ = _c("abs(v)", state={"v": "s->v"})
        assert "< 0" in expr

    def test_clip_three_args(self):
        expr, *_ = _c("clip(v, -1.0, 1.0)", state={"v": "s->v"})
        assert expr.count("?") == 2 or "if (" in expr

    def test_clip_wrong_arity_is_identity(self):
        expr, *_ = _c("clip(v)", state={"v": "s->v"})
        assert "s->v" in expr and "?" not in expr

    def test_max(self):
        expr, *_ = _c("max(v, 0.0)", state={"v": "s->v"})
        assert ">" in expr

    def test_min(self):
        expr, *_ = _c("min(v, 0.0)", state={"v": "s->v"})
        assert "<" in expr

    def test_max_single_arg_is_identity(self):
        expr, *_ = _c("max(v)", state={"v": "s->v"})
        assert "s->v" in expr

    def test_unsupported_function_raises(self):
        with pytest.raises(ValueError, match="Unsupported function"):
            _c("gamma(v)", state={"v": "s->v"})

    def test_non_name_callable_raises(self):
        with pytest.raises(ValueError, match="Only named function calls"):
            _c("m.f(v)", state={"v": "s->v"})

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            _c("exp()")


class TestLutIndexMath:
    def test_symmetric_offset_and_shift(self):
        # exp uses [-16,16) step 0.125 → offset 16<<8=4096, shift 8-3=5
        _e, stmts, *_ = _c("exp(v)", state={"v": "s->v"})
        raw = next(s for s in stmts if "_raw" in s)
        assert "4096" in raw and ">> 5" in raw

    def test_positive_log_offset_and_shift(self):
        # log uses [1/256, 8+1/256) step 1/32 → offset 1, shift 8-5=3
        _e, stmts, *_ = _c("log(v)", state={"v": "s->v"})
        raw = next(s for s in stmts if "_raw" in s)
        assert "- 1" in raw and ">> 3" in raw

    def test_lut_start_offsets_table_names(self):
        _e, _s, tables, _fv, count, _u = _c("exp(v)", state={"v": "s->v"}, lut_start=3)
        assert "_exp_lut3" in tables and count == 4


class TestRustLang:
    def test_rust_cast_syntax(self):
        expr, *_ = emit_c_fixed_expr("a + 1.0", {"a": "self.a"}, {}, Q, lang="rust")
        assert "as i64" in expr

    def test_rust_conditional_expression(self):
        expr, *_ = emit_c_fixed_expr("abs(v)", {"v": "self.v"}, {}, Q, lang="rust")
        assert "if (" in expr and "else" in expr

    def test_rust_lut_index_cast(self):
        _e, stmts, *_ = emit_c_fixed_expr("exp(v)", {"v": "self.v"}, {}, Q, lang="rust")
        assert any("let " in s and ": i64" in s for s in stmts)
        assert "as usize" in _e


class TestValidation:
    def test_invalid_lang_raises(self):
        with pytest.raises(ValueError, match="lang must be"):
            emit_c_fixed_expr("v", {"v": "s->v"}, {}, Q, lang="go")

    def test_word_too_wide_raises(self):
        with pytest.raises(ValueError, match="2\\*data_width <= 64"):
            emit_c_fixed_expr("v", {"v": "s->v"}, {}, Q88(data_width=40, fraction=8))

    def test_q16_16_supported(self):
        expr, *_ = emit_c_fixed_expr(
            "a * b", {"a": "s->a", "b": "s->b"}, {}, Q88(data_width=32, fraction=16)
        )
        assert "fxmul(" in expr

    def test_generic_visit_rejects_unknown_node(self):
        import ast

        emitter = _CFixedExprEmitter({}, {}, Q)
        with pytest.raises(ValueError, match="Unsupported AST node"):
            emitter.generic_visit(ast.parse("[1]", mode="eval").body)
