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


@pytest.mark.parametrize("expression", ("v // 0", "v // -2", "v // 3", "v // 8.0", "v // d"))
def test_floor_division_rejects_every_non_shift_divisor(expression: str) -> None:
    """No dynamic, zero, negative, fractional or non-power-of-two divider leaks in."""
    with pytest.raises(ValueError, match="Floor divisor"):
        _emit_expr(expression, _STATE_VARS, {"d": "P_D"}, Q88())


def test_floor_division_rejects_unsigned_state() -> None:
    """The exact Python negative-floor contract requires a signed datapath."""
    with pytest.raises(ValueError, match="signed fixed-point"):
        _emit_expr("v // 8", _STATE_VARS, {}, Q88(signed=False))
