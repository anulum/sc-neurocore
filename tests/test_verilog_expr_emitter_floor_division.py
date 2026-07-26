# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Verilog expression lowering contracts

"""Focused fixed-point Verilog expression lowering contracts."""

from .verilog_expr_emitter_support import *


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
