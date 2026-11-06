# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation safety numeric-limit tests

"""Text length, nested exponentiation, exponent ceiling, and literal-magnitude contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from sc_neurocore.neurons.equation_safety import (
    _DEFAULT_MAX_EXPRESSION_LENGTH,
    ExpressionSafetyValidator,
)


def test_expression_text_beyond_the_length_limit_is_refused_before_parsing() -> None:
    """A text one character over the limit is refused by length, even when it would not parse.

    The refusal names the length and the limit and does not repeat the text.
    """
    limit = _DEFAULT_MAX_EXPRESSION_LENGTH
    ExpressionSafetyValidator().validate("v" * limit)
    for text in ("v" * (limit + 1), "(" * (limit + 1)):
        with pytest.raises(
            EquationRefusal, match=f"of {limit + 1} characters exceeds limit {limit}"
        ) as refusal:
            ExpressionSafetyValidator().validate(text)
        assert len(str(refusal.value)) < 100


def test_a_validator_applies_its_own_length_limit() -> None:
    """``max_length`` sets the bound; the default applies otherwise."""
    short = ExpressionSafetyValidator(max_length=8)
    short.validate("v + 1.25")
    with pytest.raises(EquationRefusal, match="of 9 characters exceeds limit 8"):
        short.validate("v + w * 2")
    ExpressionSafetyValidator().validate("v + w * 2")


def test_a_neuron_applies_the_length_limit_before_it_parses_any_text() -> None:
    """Every public construction refuses an over-long expression by length, before any parse.

    Numbering the random draw sites is the first parse of a neuron's raw text; a
    deeply chained text that it would fail to parse is refused by length instead.
    """
    limit = _DEFAULT_MAX_EXPRESSION_LENGTH
    at_limit = "-v + 0." + "0" * (limit - len("-v + 0."))
    assert len(at_limit) == limit
    EquationNeuron({"v": at_limit}, threshold="v > 1", reset={"v": "0.0"})
    over = at_limit + "0"
    message = f"of {limit + 1} characters exceeds limit {limit}"
    with pytest.raises(EquationRefusal, match=message):
        EquationNeuron({"v": over}, threshold="v > 1")
    with pytest.raises(EquationRefusal, match=message):
        EquationNeuron({"v": "-v"}, threshold="v > 1", reset={"v": over})
    with pytest.raises(EquationRefusal, match=message):
        from_equations(f"dv/dt = {over}", threshold="v > 1")
    chained = "v*0" + "+1" * 20000
    with pytest.raises(
        EquationRefusal, match=f"of {len(chained)} characters exceeds limit {limit}"
    ):
        EquationNeuron({"v": chained}, threshold="v > 1")


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
    [
        "exp(v, w)",
        "log(v, table)",
        "clip(v, 0, 1, w)",
        "tanh(v, None, w)",
        "sin(v, [0.0])",
        "atan2(v, w, table)",
        "atan2(v, w, [0.0])",
        "atan2(v, w, None, table)",
    ],
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
        "atan2(v, w)",
        "atan2(v, w, None)",
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
