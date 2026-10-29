# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dimensional contracts of candidate namespace functions

"""Check the complete actual function namespace without invoking its numerics."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.studio.candidate_unit_expressions import UnitExpression
from sc_neurocore.studio.candidate_unit_functions import (
    DIMENSIONLESS_FUNCTIONS,
    UNIT_PRESERVING_FUNCTIONS,
    function_units,
)
from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    UnitSequence,
    coherent,
    parse_unit,
)


def test_every_actual_namespace_function_has_an_exercised_unit_rule() -> None:
    """Keep newly added numeric functions from silently bypassing unit admission."""
    other = {"sqrt", "rsqrt", "sign", "heaviside", "atan2", "hypot", "clip", "min", "max"}
    actual = {name for name, value in build_eval_namespace().items() if callable(value)}
    assert actual == DIMENSIONLESS_FUNCTIONS | UNIT_PRESERVING_FUNCTIONS | other
    expressions = UnitExpression({"v": parse_unit("mV")})
    for name in actual:
        if name in DIMENSIONLESS_FUNCTIONS:
            expression, expected = f"{name}(1)", "dimensionless"
        elif name in UNIT_PRESERVING_FUNCTIONS:
            expression, expected = f"{name}(v)", "mV"
        elif name in {"sqrt", "rsqrt"}:
            expression = f"{name}(v*v)"
            expected = "mV" if name == "sqrt" else "1/mV"
        elif name == "sign":
            expression, expected = "sign(v)", "dimensionless"
        elif name == "heaviside":
            expression, expected = "heaviside(v, 0.5)", "dimensionless"
        elif name == "atan2":
            expression, expected = "atan2(v, v)", "dimensionless"
        elif name == "hypot":
            expression, expected = "hypot(v, v)", "mV"
        elif name == "clip":
            expression, expected = "clip(v, 0, v)", "mV"
        else:
            expression, expected = f"{name}([v, 0])", "mV"
        coherent(expressions.infer(expression), parse_unit(expected))


@pytest.mark.parametrize("name", sorted(DIMENSIONLESS_FUNCTIONS))
def test_transcendental_functions_refuse_physical_arguments(name: str) -> None:
    """Require a dimensionless argument for every transcendental function.

    Parameters
    ----------
    name : str
        Function in the real numeric namespace.
    """
    expressions = UnitExpression({"v": parse_unit("mV")})
    with pytest.raises(CandidateUnitRefused, match="dimensions"):
        expressions.infer(f"{name}(v)")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("clip(v, None, v)", "mV"),
        ("clip(v, 0, None, None)", "mV"),
        ("clip([v, v], 0, v)[0]", "mV"),
        ("sin(0, None)", "dimensionless"),
        ("atan2(v, v)", "dimensionless"),
        ("max(v, 0, v)", "mV"),
        ("min([v, v])", "mV"),
        ("abs(exp([0, 1]))[0]", "dimensionless"),
        ("hypot([v, v], v)[0]", "mV"),
        ("clip(exp([[0], [1]]), 0, 1)[1][0]", "dimensionless"),
    ],
)
def test_array_functions_and_optional_arguments_keep_unit_contracts(
    expression: str, expected: str
) -> None:
    """Retain array indexing, scalar broadcasting and literal optional outputs.

    Parameters
    ----------
    expression, expected : str
        Function expression and its resulting scalar unit.
    """
    expressions = UnitExpression({"v": parse_unit("mV")})
    coherent(expressions.infer(expression), parse_unit(expected))


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("heaviside(v, v)", "dimensions"),
        ("atan2(v, w)", "dimensions"),
        ("hypot(v, w)", "dimensions"),
        ("clip(v, 0, w)", "dimensions"),
        ("clip(v, None, None)", "at least one"),
        ("clip(None, 0, v)", "scalar quantity"),
        ("min(v)", "requires a sequence"),
        ("min([])", "empty sequence"),
        ("max([[v], [v]])", "scalar quantity"),
        ("abs([v])", "plain sequence"),
        ("exp(None)", "scalar quantity"),
        ("max(exp([])) if False else 1", "empty sequence"),
    ],
)
def test_function_unit_refusals_are_authored_and_static(expression: str, fragment: str) -> None:
    """Refuse incompatible quantities without evaluating the numerical functions.

    Parameters
    ----------
    expression, fragment : str
        Invalid function expression and its authored explanation.
    """
    expressions = UnitExpression({"v": parse_unit("mV"), "w": parse_unit("pA")})
    with pytest.raises(CandidateUnitRefused, match=fragment):
        expressions.infer(expression)


def test_empty_array_propagates_and_an_unimplemented_function_refuses() -> None:
    """Handle empty shapes while refusing any future function without a unit rule."""
    result = function_units("exp", (UnitSequence(()),))
    assert isinstance(result, UnitSequence) and not result.items
    with pytest.raises(CandidateUnitRefused, match="no candidate unit rule"):
        function_units("future_namespace_function", (Unit(),))
