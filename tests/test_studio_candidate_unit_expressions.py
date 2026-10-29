# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Branch-complete static candidate unit expressions

"""Exercise quantity algebra through the public static expression admission API."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sc_neurocore.studio.candidate_unit_expressions import UnitExpression
from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    coherent,
    parse_unit,
)


@pytest.fixture
def expressions() -> UnitExpression:
    """Describe mutable physical states and immutable dimensionless parameters.

    Returns
    -------
    UnitExpression
        Admission context, with no state magnitudes or random generator.
    """
    return UnitExpression(
        {
            "v": parse_unit("mV"),
            "V": parse_unit("volt"),
            "w": parse_unit("pA"),
            "t": parse_unit("ms"),
            "x": Unit(),
            "n": Unit(constant=Fraction(2)),
            "huge": Unit(constant=Fraction(65)),
            "tiny": Unit(constant=Fraction(1, 10**300)),
        }
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("v + 0", "mV"),
        ("0 + v", "mV"),
        ("v - v", "mV"),
        ("v * 1", "mV"),
        ("v / t", "mV/ms"),
        ("v // t", "mV/ms"),
        ("v % v", "mV"),
        ("v ** n", "mV**2"),
        ("v ** (1/2)", "mV**0.5"),
        ("(v ** (1/3)) ** 3", "mV"),
        ("v ** -1", "1/mV"),
        ("x ** x", "dimensionless"),
        ("-v", "mV"),
        ("+v", "mV"),
        ("not x", "dimensionless"),
        ("True and False or x", "dimensionless"),
        ("0 <= v <= v", "dimensionless"),
        ("v if x else 0", "mV"),
        ("v if True else v", "mV"),
        ("[v, w][0]", "mV"),
        ("(v, w)[-1]", "pA"),
        ("min([v, v][0:2:1])", "mV"),
        ("max((v, v)[::-1])", "mV"),
        ("min([v, v][:])", "mV"),
        ("min([v, v][+0:])", "mV"),
        ("v.real", "mV"),
        ("v.imag", "mV"),
        ("sqrt([v*v, v*v])[0]", "mV"),
        ("max(exp([x, 1]))", "dimensionless"),
        ("min(exp([x, 1]) + exp([1]))", "dimensionless"),
        ("(-exp([0, 1]))[0]", "dimensionless"),
        ("(+exp([0, 1]))[0]", "dimensionless"),
        ("exp([x, 1]).real[0]", "dimensionless"),
        ("([v, v] if x else [v, 0])[0]", "mV"),
        ("(exp([x, 1]) if x else exp([0, 1]))[0]", "dimensionless"),
        ("v ** (n - 1)", "mV"),
        ("v ** (n % 2)", "dimensionless"),
        ("v ** (n // 2)", "mV"),
        ("v ** (n * 1)", "mV**2"),
        ("v ** (+n / 2)", "mV"),
        ("n ** 2", "dimensionless"),
        ("n ** (1/2)", "dimensionless"),
        ("tiny ** -64", "dimensionless"),
    ],
)
def test_checked_expressions_preserve_physical_dimensions(
    expressions: UnitExpression, expression: str, expected: str
) -> None:
    """Check scalar, array, selection and exact rational-power paths.

    Parameters
    ----------
    expressions : UnitExpression
        Static quantity declarations.
    expression, expected : str
        Authored expression and its physical result unit.
    """
    result = expressions.infer(expression)
    unit = parse_unit(expected)
    coherent(result, unit)
    assert result.dimensions == unit.dimensions
    assert result.scale == pytest.approx(unit.scale, rel=1e-12)


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("v + w", "dimensions"),
        ("v + V", "unit scales"),
        ("v % t", "dimensions"),
        ("v if True else w", "dimensions"),
        ("v if v else v", "dimensions"),
        ("True or v", "dimensions"),
        ("False and v", "dimensions"),
        ("not v", "dimensions"),
        ("v < v < w", "dimensions"),
        ("v ** v", "dimensions"),
        ("v ** x", "static dimensionless exponent"),
        ("v ** huge", "static dimensionless exponent"),
        ("v ** (1/0)", "static dimensionless exponent"),
        ("v ** (n * 1e15)", "static dimensionless exponent"),
        ("exp", "function cannot be used"),
        ("None", "scalar quantity"),
        ("1j", "real scalar literal"),
        ("[v]", "scalar quantity"),
        ("(v if x else [v])", "scalar quantity"),
        ("([v] if x else [v,v])[0]", "same quantity shape"),
        ("[v].real", "plain sequence"),
        ("-[v]", "plain sequences"),
        ("([v] if x else [v]) * 2", "plain sequences"),
        ("v[0]", "only a sequence"),
        ("[v][1]", "outside its sequence"),
        ("[v][None]", "literal integer"),
        ("[v][True]", "literal integer"),
        ("[v][x]", "literal integer"),
        ("[v][::0]", "slice step"),
        ("[v][:v]", "literal integer"),
        ("min([v, exp(w)])", "dimensions"),
        ("exp([x, v])[0]", "dimensions"),
        ("exp([0,1]) + exp([0,1,2])", "shapes must agree"),
    ],
)
def test_unsafe_unit_paths_refuse_without_evaluating_runtime_values(
    expressions: UnitExpression, expression: str, fragment: str
) -> None:
    """Require a known coherent unit even when a branch is unreachable initially.

    Parameters
    ----------
    expressions : UnitExpression
        Static quantity declarations without mutable magnitudes.
    expression, fragment : str
        Authored invalid expression and its safe refusal explanation.
    """
    with pytest.raises(CandidateUnitRefused, match=fragment):
        expressions.infer(expression)
