# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Static dimensions through all candidate expression branches

"""Inspect the checked equation AST without running a candidate or drawing noise.

Both sides of every conditional, Boolean choice and comparison chain are
inspected. Constants are propagated only through bounded scalar arithmetic to
establish exponents; mutable state and inputs never acquire constant values.
The numerical equation program, evaluation order and namespace are untouched.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import cast

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator
from sc_neurocore.studio.candidate_expressions import validate_candidate_calls
from sc_neurocore.studio.candidate_unit_functions import elementwise, function_units
from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    UnitSequence,
    UnitValue,
    coherent,
    dimensionless,
    product,
    scalar,
)

_CONSTANT_OPERATORS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}


def _constant(operation: ast.operator, first: Unit, second: Unit) -> Fraction | None:
    """Fold bounded dimensionless constants solely to establish static exponents."""
    if first.dimensions or second.dimensions or first.scale != 1.0 or second.scale != 1.0:
        return None
    if first.constant is None or second.constant is None:
        return None
    if isinstance(operation, ast.Pow):
        exponent = second.constant
        if (
            exponent.denominator != 1
            or abs(exponent) > 64
            or max(first.constant.numerator.bit_length(), first.constant.denominator.bit_length())
            * abs(exponent)
            > 4096
        ):
            return None
    try:
        result = _CONSTANT_OPERATORS[type(operation)](first.constant, second.constant)
    except (ArithmeticError, ValueError):
        return None
    return Fraction(result) if abs(result) <= 1e15 else None


def _binary(operation: ast.operator, first: Unit, second: Unit) -> Unit:
    """Combine scalar units using the checked numeric operator's dimensional rule."""
    if isinstance(operation, (ast.Add, ast.Sub, ast.Mod)):
        result = coherent(first, second)
    elif isinstance(operation, (ast.Mult, ast.Div, ast.FloorDiv)):
        result = product(first, second, divide=not isinstance(operation, ast.Mult))
    else:
        dimensionless(second)
        if not first.dimensions and first.scale == 1.0:
            result = Unit()
        else:
            if (
                second.constant is None
                or not math.isfinite(second.constant)
                or abs(second.constant) > 64
            ):
                raise CandidateUnitRefused(
                    "a unitful base requires a static dimensionless exponent from -64 to 64"
                )
            result = first.power(second.constant)
    return Unit(
        result.dimensions,
        result.scale,
        _constant(operation, first, second),
        first.literal and second.literal,
    )


def _join(first: UnitValue, second: UnitValue) -> UnitValue:
    """Require matching shapes and units across both conditional branches."""
    if isinstance(first, UnitSequence) and isinstance(second, UnitSequence):
        if len(first.items) != len(second.items):
            raise CandidateUnitRefused("conditional branches must have the same quantity shape")
        return UnitSequence(
            tuple(_join(a, b) for a, b in zip(first.items, second.items, strict=True)),
            first.array and second.array,
        )
    return coherent(scalar(first), scalar(second))


def _index(node: ast.expr | None) -> int | None:
    """Read a literal integer selection without evaluating a state-dependent index."""
    if node is None:
        return None
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return sign * node.value
    raise CandidateUnitRefused(
        "a sequence index or slice requires a literal integer to establish its units"
    )


class UnitExpression:
    """Infer units using an immutable snapshot of the admitted symbol descriptions.

    Parameters
    ----------
    symbols : mapping
        State, parameter and input units, and dimensionless namespace constants.
        Function names are handled only as callees, never as numeric quantities.
    """

    def __init__(self, symbols: Mapping[str, Unit]) -> None:
        """Copy the symbol declarations so later caller mutations cannot alter admission."""
        self._symbols = dict(symbols)

    def infer(self, expression: str) -> Unit:
        """Require a scalar result after checking every expression path.

        Parameters
        ----------
        expression : str
            Authored DSL expression; the existing safety gate remains in force.

        Returns
        -------
        Unit
            The inferred scalar dimensions and scale.

        Raises
        ------
        CandidateUnitRefused
            A path has incompatible dimensions, scales or quantity shapes, or
            its units cannot be established without a runtime value.
        EquationRefusal
            The existing equation safety gate refuses the expression syntax.
        """
        tree = ExpressionSafetyValidator().checked_tree(expression)
        validate_candidate_calls(tree)
        return scalar(self._infer(tree.body))

    def _infer(self, node: ast.expr) -> UnitValue:
        """Inspect every child node, including unselected branches and sequence elements."""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return None
            if isinstance(node.value, (int, float, bool)):
                return Unit(
                    constant=Fraction(
                        str(int(node.value) if isinstance(node.value, bool) else node.value)
                    ),
                    literal=True,
                )
            raise CandidateUnitRefused("a real scalar literal is required to establish units")
        if isinstance(node, ast.Name):
            if node.id not in self._symbols:
                raise CandidateUnitRefused("an equation function cannot be used as a quantity")
            return self._symbols[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return UnitSequence(tuple(self._infer(item) for item in node.elts))
        if isinstance(node, ast.BinOp):
            first, second = self._infer(node.left), self._infer(node.right)
            if any(
                isinstance(value, UnitSequence) and not value.array for value in (first, second)
            ):
                raise CandidateUnitRefused(
                    "plain sequences cannot supply numeric equation arithmetic"
                )
            return elementwise((first, second), lambda values: _binary(node.op, *values))
        if isinstance(node, ast.UnaryOp):
            value = self._infer(node.operand)
            if isinstance(node.op, ast.Not):
                dimensionless(scalar(value))
                return Unit()
            if isinstance(value, UnitSequence) and not value.array:
                raise CandidateUnitRefused("plain sequences cannot supply a numeric unary operand")

            def signed(values: tuple[Unit, ...]) -> Unit:
                """Retain units and bounded static constants under unary signs."""
                item = values[0]
                constant = item.constant
                if constant is not None and isinstance(node.op, ast.USub):
                    constant = -constant
                return Unit(item.dimensions, item.scale, constant, item.literal)

            return elementwise((value,), signed)
        if isinstance(node, ast.Compare):
            values = [self._infer(node.left), *(self._infer(item) for item in node.comparators)]
            for first, second in zip(values, values[1:]):
                coherent(scalar(first), scalar(second))
            return Unit()
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                dimensionless(scalar(self._infer(item)))
            return Unit()
        if isinstance(node, ast.IfExp):
            dimensionless(scalar(self._infer(node.test)))
            first, second = self._infer(node.body), self._infer(node.orelse)
            return _join(first, second)
        if isinstance(node, ast.Call):
            name = cast(ast.Name, node.func).id
            return function_units(name, tuple(self._infer(item) for item in node.args))
        if isinstance(node, ast.Attribute):
            value = self._infer(node.value)
            if isinstance(value, UnitSequence) and not value.array:
                raise CandidateUnitRefused(
                    "a plain sequence has no scalar real or imaginary quantity"
                )
            return elementwise((value,), lambda values: Unit(values[0].dimensions, values[0].scale))
        # The closed safety grammar leaves only Subscript after these cases;
        # Slice nodes are visited only as its index, never as a value.
        subscript = cast(ast.Subscript, node)
        value = self._infer(subscript.value)
        if not isinstance(value, UnitSequence):
            raise CandidateUnitRefused("only a sequence can supply an indexed quantity")
        if isinstance(subscript.slice, ast.Slice):
            indices = [
                _index(part)
                for part in (subscript.slice.lower, subscript.slice.upper, subscript.slice.step)
            ]
            if indices[2] == 0:
                raise CandidateUnitRefused("a sequence slice step must be nonzero")
            return UnitSequence(value.items[slice(*indices)], value.array)
        index = _index(subscript.slice)
        if index is None or not -len(value.items) <= index < len(value.items):
            raise CandidateUnitRefused("a quantity index is outside its sequence")
        return value.items[index]
