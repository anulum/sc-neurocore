# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiled equation value error boundary

"""Separate expected expression-value failures from errors in the engine itself.

Only the equation program and its existing float/bool conversion run inside
this boundary. State updates, RNG handling, storage and integration bookkeeping
remain outside it. Expected value failures preserve their standard exception
categories and carry fixed authored reasons; their generated causes stay private.
Existing authored domain guards pass through unchanged. No equation, conversion,
namespace or numerical operation is substituted. A program is a translation of
a checked expression tree into plain function calls
(:mod:`sc_neurocore.neurons.equation_program`); this module runs no dynamic
code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import SupportsFloat, SupportsIndex, TypeVar

from sc_neurocore.neurons.equation_program import EquationProgram
from sc_neurocore.neurons.equation_refusals import (
    EquationAttributeFailure,
    EquationDivisionFailure,
    EquationEvaluationFailure,
    EquationFloatingFailure,
    EquationIndexFailure,
    EquationNameFailure,
    EquationOverflowFailure,
    EquationRefusal,
    EquationTypeFailure,
)


_Converted = TypeVar("_Converted")
_ScalarInput = str | bytes | bytearray | SupportsFloat | SupportsIndex


def evaluate_scalar(
    program: EquationProgram,
    environment: Mapping[str, object],
) -> float:
    """Run a checked expression program with its existing float conversion.

    Expected value failures raise EquationEvaluationFailure subclasses while
    retaining their standard exception categories; other faults propagate.

    Parameters
    ----------
    program:
        Program translated from an expression that ExpressionSafetyValidator checked.
    environment:
        Current numerical bindings, passed to the expression unchanged.

    Returns
    -------
    float
        The expression value after the existing scalar conversion.

    Raises
    ------
    EquationEvaluationFailure
        If evaluation or conversion fails in an expected value category.
    EquationRefusal
        If an existing authored function-domain guard refuses the value.
    """
    return _evaluate_value(program, environment, float)


def evaluate_condition(
    program: EquationProgram,
    environment: Mapping[str, object],
) -> bool:
    """Run a checked threshold program with its existing bool conversion.

    The expression and its conversion share the scalar value error boundary.
    No state or random stream is advanced by this helper.

    Parameters
    ----------
    program:
        Program translated from a threshold that ExpressionSafetyValidator checked.
    environment:
        Current numerical bindings, passed to the expression unchanged.

    Returns
    -------
    bool
        The expression value after the existing threshold conversion.

    Raises
    ------
    EquationEvaluationFailure
        If evaluation or conversion fails in an expected value category.
    EquationRefusal
        If an existing authored function-domain guard refuses the value.
    """
    return _evaluate_value(program, environment, bool)


def _evaluate_value(
    program: EquationProgram,
    environment: Mapping[str, object],
    convert: Callable[[_ScalarInput], _Converted],
) -> _Converted:
    """Run a checked expression program and mark expected value failures.

    A program exists only for an expression that passed
    ExpressionSafetyValidator, because the translator takes the tree from the
    gate itself. Exceptions outside the listed value categories are propagated
    without a public marker.

    Parameters
    ----------
    program:
        Program translated from an expression that passed the safety gate.
    environment:
        The caller's current numerical namespace, inputs and state.
    convert:
        The caller's existing float or bool conversion.

    Returns
    -------
    float or bool
        The expression's scalar value or threshold decision.

    Raises
    ------
    EquationEvaluationFailure
        When expression evaluation or conversion raises an expected value
        failure; subclasses also preserve its standard exception category.
    EquationRefusal
        When an existing authored function-domain guard refuses the value.
    """
    try:
        return convert(program(environment))
    except EquationRefusal:
        raise
    except ZeroDivisionError as exc:
        raise EquationDivisionFailure("the equation divides by zero") from exc
    except OverflowError as exc:
        raise EquationOverflowFailure("the equation value exceeds its numeric range") from exc
    except FloatingPointError as exc:
        raise EquationFloatingFailure("the equation arithmetic could not remain finite") from exc
    except TypeError as exc:
        raise EquationTypeFailure("the equation value has an invalid type") from exc
    except IndexError as exc:
        raise EquationIndexFailure("the equation index is outside its value") from exc
    except AttributeError as exc:
        raise EquationAttributeFailure("the equation reads an unavailable attribute") from exc
    except NameError as exc:
        raise EquationNameFailure("the equation reads an unavailable symbol") from exc
    except ValueError as exc:
        raise EquationEvaluationFailure("the equation value is invalid") from exc
