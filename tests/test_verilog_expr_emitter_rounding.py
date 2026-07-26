# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Verilog expression lowering contracts

"""Focused fixed-point Verilog expression lowering contracts."""

from .verilog_expr_emitter_support import *


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
