# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Verilog expression lowering contracts

"""Focused fixed-point Verilog expression lowering contracts."""

from .verilog_expr_emitter_support import *


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


def test_cosh_and_exprel_calls_emit_dedicated_luts() -> None:
    """Hyperbolic and exponential-relative calls use their own LUT contracts."""
    cosh_result, cosh_intermediates, *_ = _emit_expr("cosh(v)", _STATE_VARS, {}, Q88())
    exprel_result, exprel_intermediates, *_ = _emit_expr("exprel(v)", _STATE_VARS, {}, Q88())

    assert cosh_result.startswith("_cosh_lut")
    assert any("_cosh_lut" in line for line in cosh_intermediates)
    assert exprel_result.startswith("_exprel_lut")
    assert any("_exprel_lut" in line for line in exprel_intermediates)
