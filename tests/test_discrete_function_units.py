# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Discontinuous fixed-point function references
"""Exercise rounding and Heaviside through the public function-unit dispatcher."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sc_neurocore.compiler.function_units import evaluate


@pytest.mark.parametrize(
    ("width", "fraction"), [(2, 0), (8, 4), (8, 7), (16, 8), (16, 12), (32, 16), (32, 24)]
)
def test_round_codes_match_exact_rational_rounding(width: int, fraction: int) -> None:
    """All narrow words and wide tie/endpoints match independent rational ties-to-even."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    scale = 1 << fraction
    codes: range | list[int]
    if width <= 16:
        codes = range(low, high + 1)
    else:
        candidates = {low, high, -1, 0, 1}
        for integer in (-1000, -3, -2, -1, 0, 1, 2, 3, 1000):
            candidates.update(integer * scale + scale // 2 + delta for delta in (-1, 0, 1))
        codes = sorted(code for code in candidates if low <= code <= high)
    for code in codes:
        expected = max(low, min(high, round(Fraction(code, scale)) * scale))
        assert evaluate("round", code, width, fraction) == expected


@pytest.mark.parametrize(
    ("width", "fraction"), [(2, 0), (8, 4), (8, 7), (16, 8), (16, 12), (32, 16), (32, 24)]
)
def test_heaviside_preserves_sign_and_dynamic_zero_value(width: int, fraction: int) -> None:
    """Every sign branch retains the supplied zero value and saturates positive one."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    for code in (low, -1, 0, 1, high):
        for at_zero in (low, -1, 0, 1, high):
            expected = 0 if code < 0 else at_zero if code == 0 else min(1 << fraction, high)
            assert (
                evaluate("heaviside", code, width, fraction, additional_inputs=(at_zero,))
                == expected
            )


@pytest.mark.parametrize(("width", "fraction"), [(8, 7), (16, 8), (16, 12)])
def test_heaviside_zero_accepts_every_encoded_second_operand(width: int, fraction: int) -> None:
    """At zero the whole second-operand word survives, including both signed extremes."""
    for at_zero in range(-(1 << (width - 1)), 1 << (width - 1)):
        assert evaluate("heaviside", 0, width, fraction, additional_inputs=(at_zero,)) == at_zero


@pytest.mark.parametrize("function", ["round", "heaviside"])
def test_discrete_units_refuse_outside_word_codes(function: str) -> None:
    """Invalid first or second words cannot silently enter a declared eight-bit domain."""
    for code in (-129, 128):
        additional = (0,) if function == "heaviside" else ()
        with pytest.raises(ValueError, match="outside signed 8-bit domain"):
            evaluate(function, code, 8, 4, additional_inputs=additional)
        if function == "heaviside":
            with pytest.raises(ValueError, match="outside signed 8-bit domain"):
                evaluate(function, 0, 8, 4, additional_inputs=(code,))


@pytest.mark.parametrize(
    ("function", "additional"), [("round", (1,)), ("heaviside", ()), ("heaviside", (1, 2))]
)
def test_discrete_units_require_exact_operand_counts(
    function: str, additional: tuple[int, ...]
) -> None:
    """The registry admits exactly one rounding operand and two Heaviside operands."""
    with pytest.raises(ValueError, match="needs exactly"):
        evaluate(function, 0, 16, 8, additional_inputs=additional)
