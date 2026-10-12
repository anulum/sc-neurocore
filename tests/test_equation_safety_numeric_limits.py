# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety numeric-limit tests

"""Nested exponentiation, exponent ceiling, and literal-magnitude contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator


def test_nested_exponentiation_is_rejected() -> None:
    """Chained exponents (``9**9**9**9``) blow up under eval and are refused."""
    with pytest.raises(ValueError, match="Nested exponentiation blocked"):
        ExpressionSafetyValidator().validate("9**9**9**9")


def test_oversized_literal_exponent_is_rejected() -> None:
    """A single ``**`` with a huge literal exponent is refused before eval."""
    with pytest.raises(ValueError, match="Exponent .* exceeds limit"):
        ExpressionSafetyValidator().validate("9 ** 999999")


def test_oversized_numeric_constant_is_rejected() -> None:
    """A literal whose magnitude overflows the sandbox ceiling is refused."""
    with pytest.raises(ValueError, match="exceeds magnitude limit"):
        ExpressionSafetyValidator().validate("1e400")


@pytest.mark.parametrize(
    "expression",
    ["exp(v, w)", "log(v, table)", "clip(v, 0, 1, w)", "tanh(v, None, w)", "sin(v, [0.0])"],
)
def test_an_array_function_cannot_be_given_an_output_argument(expression: str) -> None:
    """No value but the literal ``None`` reaches the output-array parameter of an array function.

    Parameters
    ----------
    expression : str
        Call that passes a value after the inputs of a namespace array function.
    """
    with pytest.raises(ValueError, match="cannot be given an output argument"):
        ExpressionSafetyValidator().validate(expression)
    validator = ExpressionSafetyValidator()
    for accepted in (
        "clip(exp(v), 0, max(v, w, 1))",
        "exp(1, None)",
        "clip(v, None, 0, None)",
        "clip(v)",
        "user_function(v, w, 1)",
        "sqrt(v, w)",
    ):
        validator.validate(accepted)


@pytest.mark.parametrize(
    "expression", ["[0.0] * 4", "n * (v,)", "(v,) + (w,)", "[v] + table", "table + [v]"]
)
def test_a_sequence_literal_cannot_be_repeated_or_concatenated(expression: str) -> None:
    """Repetition and concatenation of a literal sequence are refused.

    Parameters
    ----------
    expression : str
        Expression that would build a sequence whose size the expression chooses.
    """
    with pytest.raises(ValueError, match="cannot be repeated or concatenated"):
        ExpressionSafetyValidator().validate(expression)
    ExpressionSafetyValidator().validate("min([v, 0]) + (v, w)[0] * 2")
