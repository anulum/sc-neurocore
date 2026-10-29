# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unit rules for the actual candidate equation functions

"""Infer equation-function units, including every array input and optional out.

These are admission rules, not replacement numerical implementations. The
runtime namespace and its binary64 functions remain unchanged. All sequence
elements are inspected; no initial-state value or conditional selects a path.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    UnitSequence,
    UnitValue,
    coherent,
    dimensionless,
    scalar,
)

DIMENSIONLESS_FUNCTIONS = frozenset(
    {
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
        "cosh",
        "sinh",
        "tan",
        "arctan",
        "arcsin",
        "arccos",
        "exp2",
        "log2",
        "log10",
        "expm1",
        "log1p",
        "softplus",
        "erf",
        "exprel",
        "sigmoid",
        "expit",
    }
)
"""Functions whose inputs and results must be unscaled dimensionless values."""
UNIT_PRESERVING_FUNCTIONS = frozenset({"abs", "floor", "ceil", "trunc", "round"})
"""Numeric rounding is in the explicitly declared unit; no conversion is made."""


def elementwise(
    arguments: tuple[UnitValue, ...], rule: Callable[[tuple[Unit, ...]], Unit]
) -> UnitValue:
    """Apply a scalar unit rule to every element of broadcastable arrays.

    Parameters
    ----------
    arguments : tuple
        Scalar or sequence unit descriptions; None is not an input quantity.
    rule : callable
        Scalar unit rule, which never evaluates numerical model expressions.

    Returns
    -------
    UnitValue
        A scalar or an array of inferred units.

    Raises
    ------
    CandidateUnitRefused
        An input is not a quantity, shapes cannot broadcast, or a scalar rule
        finds incompatible dimensions or scales.
    """
    sequences = [value for value in arguments if isinstance(value, UnitSequence)]
    if not sequences:
        return rule(tuple(scalar(value) for value in arguments))
    lengths = {len(value.items) for value in sequences}
    non_singleton = lengths - {1}
    if len(non_singleton) > 1:
        raise CandidateUnitRefused("array shapes must agree to establish the expression's units")
    extent = next(iter(non_singleton), 1)
    return UnitSequence(
        tuple(
            elementwise(
                tuple(
                    value.items[0 if len(value.items) == 1 else index]
                    if isinstance(value, UnitSequence)
                    else value
                    for value in arguments
                ),
                rule,
            )
            for index in range(extent)
        ),
        array=True,
    )


def _common(arguments: tuple[Unit, ...]) -> Unit:
    """Require one coherent unit for all extrema or clipping operands."""
    if not arguments:
        raise CandidateUnitRefused("an empty sequence cannot establish an extremum's units")
    result = arguments[0]
    for value in arguments[1:]:
        result = coherent(result, value)
    return result


def _scalar_call(name: str, arguments: tuple[Unit, ...]) -> Unit:
    """Apply a namespace function's scalar dimensional rule without calling it."""
    first = arguments[0]
    if name in DIMENSIONLESS_FUNCTIONS:
        dimensionless(first)
        return Unit()
    if name in UNIT_PRESERVING_FUNCTIONS:
        return Unit(first.dimensions, first.scale)
    if name in {"sqrt", "rsqrt"}:
        return first.power(Fraction(1 if name == "sqrt" else -1, 2))
    if name == "sign":
        return Unit()
    if name == "heaviside":
        dimensionless(arguments[1])
        return Unit()
    if name == "atan2":
        coherent(first, arguments[1])
        return Unit()
    if name == "hypot":
        return coherent(first, arguments[1])
    if name == "clip":
        return _common(arguments)
    raise CandidateUnitRefused("the equation function has no candidate unit rule")


def function_units(name: str, arguments: tuple[UnitValue, ...]) -> UnitValue:
    """Infer an admitted namespace call without invoking its numerical function.

    Parameters
    ----------
    name : str
        Direct callee already admitted by the equation safety and arity gates.
    arguments : tuple
        All statically inspected argument descriptions, including literal None
        in permitted optional output and clipping-bound positions.

    Returns
    -------
    UnitValue
        Inferred result, retaining array shape until an index or reduction.

    Raises
    ------
    CandidateUnitRefused
        A quantity, function, iterable or scale cannot satisfy the unit rule.
    """
    if name in {"min", "max"}:
        if len(arguments) == 1:
            sequence = arguments[0]
            if not isinstance(sequence, UnitSequence):
                raise CandidateUnitRefused(
                    "a single-argument extremum requires a sequence of quantities"
                )
            arguments = sequence.items
        return _common(tuple(scalar(value) for value in arguments))
    if name == "clip":
        inputs = arguments[:3]
        if inputs[1] is None and inputs[2] is None:
            raise CandidateUnitRefused("clip requires at least one quantity bound")
        active: tuple[UnitValue, ...] = tuple(value for value in inputs if value is not None)
        # The first position cannot be omitted: None there is a failed input,
        # not an absent bound or optional output.
        if not isinstance(inputs[0], UnitSequence):
            scalar(inputs[0])
    else:
        count = 2 if name in {"atan2", "hypot", "heaviside"} else 1
        active = arguments[:count]
    if name == "abs" and isinstance(active[0], UnitSequence) and not active[0].array:
        raise CandidateUnitRefused("abs requires a numeric quantity rather than a plain sequence")
    return elementwise(active, lambda values: _scalar_call(name, values))
