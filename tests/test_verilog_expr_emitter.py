# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog expression emitter tests

"""Focused contracts for fixed-point Verilog expression lowering."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr

_STATE_VARS = {"v": "v"}


def test_fractional_power_expressions_emit_root_luts() -> None:
    """Fractional powers lower to bounded root lookup tables."""
    sqrt_result, sqrt_intermediates, *_ = _emit_expr("v ** 0.5", _STATE_VARS, {}, Q88())
    cbrt_result, cbrt_intermediates, *_ = _emit_expr("v ** (1.0 / 3.0)", _STATE_VARS, {}, Q88())

    assert sqrt_result.startswith("_sqrt_lut")
    assert any("_sqrt_lut" in line for line in sqrt_intermediates)
    assert cbrt_result.startswith("_cbrt_lut")
    assert any("_cbrt_lut" in line for line in cbrt_intermediates)


@pytest.mark.parametrize("expression", ["sqrt(v)", "v ** 0.5"])
def test_sqrt_lowering_uses_the_shared_non_negative_half_unit_grid(
    expression: str,
) -> None:
    """Function and power syntax index the same table at zero, one-half, and one."""
    _result, intermediates, *_ = _emit_expr(expression, _STATE_VARS, {}, Q88())
    joined = "\n".join(intermediates)

    assert "16 entries over [0.0, 8.0), step 0.5" in joined
    assert "17'sd0) >>> 7" in joined
    assert "4'd0: _sqrt_lut0_out = 16'sd0;" in joined
    assert "4'd2: _sqrt_lut0_out = 16'sd256;" in joined
    assert "4'd8: _sqrt_lut0_out = 16'sd512;" in joined


def test_nearest_rounding_uses_a_sign_aware_half_lsb_bias() -> None:
    """Negative half ties receive half-minus-one before the arithmetic shift."""
    _result, intermediates, *_ = _emit_expr(
        "v * 0.5",
        _STATE_VARS,
        {},
        Q88(rounding="nearest"),
    )
    joined = "\n".join(intermediates)

    assert "_mul0[31] ? 32'sd127 : 32'sd128" in joined
    assert "_rnd_half1 = _mul0 + _rnd_bias1" in joined
    assert "_lfsr" not in joined


def test_bankers_rounding_emits_tie_to_even_guard() -> None:
    """Banker's rounding detects exact half ties and clears an odd result bit."""
    _result, intermediates, *_ = _emit_expr(
        "v * 0.5",
        _STATE_VARS,
        {},
        Q88(rounding="bankers"),
    )
    joined = "\n".join(intermediates)

    assert "_rnd_biased1 = _mul0 + 128" in joined
    assert "_rnd_guard1 = (_mul0[7:0] == 128)" in joined
    assert "~16'd1" in joined


@pytest.mark.parametrize("rounding", ["nearest", "bankers"])
def test_fraction_zero_rounding_directly_narrows_without_negative_indices(
    rounding: str,
) -> None:
    """Integer-only formats emit no half-bit shift or negative part-select."""
    result, intermediates, *_ = _emit_expr(
        "v * 2",
        _STATE_VARS,
        {},
        Q88(data_width=32, fraction=0, rounding=rounding),
    )

    assert result == "_t0"
    assert intermediates[-1] == "wire signed [31:0] _t0 = _mul0;"
    assert "-1" not in intermediates[-1]


def test_stochastic_rounding_without_an_owned_lfsr_fails_closed() -> None:
    """Direct expression lowering cannot emit an undeclared ``_lfsr`` reference."""
    with pytest.raises(NotImplementedError, match="caller-owned LFSR"):
        _emit_expr("v * 0.5", _STATE_VARS, {}, Q88(rounding="stochastic"))


def test_cosh_and_exprel_calls_emit_dedicated_luts() -> None:
    """Hyperbolic and exponential-relative calls use their own LUT contracts."""
    cosh_result, cosh_intermediates, *_ = _emit_expr("cosh(v)", _STATE_VARS, {}, Q88())
    exprel_result, exprel_intermediates, *_ = _emit_expr("exprel(v)", _STATE_VARS, {}, Q88())

    assert cosh_result.startswith("_cosh_lut")
    assert any("_cosh_lut" in line for line in cosh_intermediates)
    assert exprel_result.startswith("_exprel_lut")
    assert any("_exprel_lut" in line for line in exprel_intermediates)


def test_unsupported_unary_operator_fails_closed() -> None:
    """Unsupported unary operators are rejected instead of lowered silently."""
    with pytest.raises(ValueError, match="Unsupported unary op"):
        _emit_expr("~v", _STATE_VARS, {}, Q88())


def test_unsupported_ast_node_fails_closed() -> None:
    """Unsupported AST nodes are rejected by the generic visitor."""
    with pytest.raises(ValueError, match="Unsupported AST node"):
        _emit_expr("[v]", _STATE_VARS, {}, Q88())


def test_floor_division_emits_signed_python_floor_and_restores_q_scale() -> None:
    """Q16.8 ``-10 // 8`` lowers to arithmetic -2 represented as -512."""
    result, intermediates, *_ = _emit_expr("v // 8", _STATE_VARS, {}, Q88())

    assert result == "_floordiv0"
    assert intermediates == [
        "wire signed [15:0] _floordiv0_dividend = v_reg;",
        "wire signed [15:0] _floordiv0_integer = $signed(_floordiv0_dividend) >>> 11;",
        "wire signed [15:0] _floordiv0 = _floordiv0_integer <<< 8;",
    ]


def test_q320_floor_division_is_the_iqif_arithmetic_shift() -> None:
    """An integer-only Q32.0 path lowers ``force // 8`` to ``force >>> 3``."""
    q320 = Q88(data_width=32, fraction=0)
    result, intermediates, *_ = _emit_expr("v // 8", _STATE_VARS, {}, q320)

    assert result == "_floordiv0"
    assert "$signed(_floordiv0_dividend) >>> 3" in intermediates[1]
    assert intermediates[2].endswith("= _floordiv0_integer;")


@pytest.mark.parametrize(
    "expression",
    ("v // 0", "v // -2", "v // 3", "v // 8.0", "v // 128", "v // d"),
)
def test_floor_division_rejects_every_non_shift_divisor(expression: str) -> None:
    """No dynamic, zero, negative, fractional or non-power-of-two divider leaks in."""
    with pytest.raises(ValueError, match="Floor divisor"):
        _emit_expr(expression, _STATE_VARS, {"d": "P_D"}, Q88())


def test_floor_division_rejects_unsigned_state() -> None:
    """The exact Python negative-floor contract requires a signed datapath."""
    with pytest.raises(ValueError, match="signed fixed-point"):
        _emit_expr("v // 8", _STATE_VARS, {}, Q88(signed=False))


@pytest.mark.parametrize(
    ("expression", "q", "message"),
    [
        ("v % 128.0", Q88(), "exceeds fixed-point maximum"),
        ("v % 0.001", Q88(), "underflows"),
    ],
)
def test_modulo_rejects_unrepresentable_or_unsigned_contracts(
    expression: str,
    q: Q88,
    message: str,
) -> None:
    """Modulo requires signed state and a positive representable literal period."""
    with pytest.raises(ValueError, match=message):
        _emit_expr(expression, _STATE_VARS, {}, q)


def test_modulo_rejects_unsigned_state_before_divisor_validation() -> None:
    """Unsigned state fails at the modulo contract even for a named period."""
    with pytest.raises(ValueError, match="requires signed"):
        _emit_expr(
            "v % period",
            _STATE_VARS,
            {"period": "P_PERIOD"},
            Q88(signed=False),
        )


def test_if_expression_lowers_to_a_verilog_ternary() -> None:
    """Piecewise expressions retain their comparison, true branch, and false branch."""
    result, *_ = _emit_expr("v if v > 0.0 else -v", _STATE_VARS, {}, Q88())

    assert result.startswith("(((v_reg >")
    assert ") ? (v_reg) : ((-v_reg)))" in result
