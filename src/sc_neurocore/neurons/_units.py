# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional pint helpers for strict dimensional checking

"""Check equation expressions for dimensional consistency with pint quantities.

Strict-units mode evaluates each authored expression once with unit-carrying
values and compares the unit of the result with the unit the equation must
have. The expression is run as a checked program of plain function calls
(:mod:`sc_neurocore.neurons.equation_program`), never by the interpreter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.exceptions import SCDependencyError
from sc_neurocore.neurons.equation_program import compile_expression

try:
    import pint

    HAS_PINT = True
    UNIT_REGISTRY: Any = pint.UnitRegistry()
    _DIMENSIONAL_ERROR: type[Exception] = pint.DimensionalityError
except ImportError:  # pragma: no cover - exercised via dependency error path
    pint = None  # type: ignore[assignment]
    HAS_PINT = False
    UNIT_REGISTRY = None

    class _FallbackDimensionalError(ValueError):
        """Fallback dimensional error when pint is unavailable."""

    _DIMENSIONAL_ERROR = _FallbackDimensionalError


DimensionalError = _DIMENSIONAL_ERROR


def require_pint() -> None:
    """Refuse strict-units work when the optional pint package is missing.

    Raises
    ------
    SCDependencyError
        pint is not installed.
    """
    if not HAS_PINT:
        raise SCDependencyError(
            "pint is not available. Install with: pip install sc-neurocore[units]"
        )


def is_quantity(value: Any) -> bool:
    """Tell whether a value is a pint quantity.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True only if pint is installed and the value is one of its quantities.
    """
    return HAS_PINT and isinstance(value, pint.Quantity)


def require_quantity(value: Any, label: str) -> Any:
    """Return a value unchanged if it is a pint quantity, and refuse it otherwise.

    Parameters
    ----------
    value : Any
        Value that strict-units mode requires to carry a unit.
    label : str
        Name used in the refusal.

    Returns
    -------
    Any
        The same quantity.

    Raises
    ------
    SCDependencyError
        pint is not installed.
    ValueError
        The value carries no unit.
    """
    require_pint()
    if not is_quantity(value):
        raise ValueError(f"{label} must be a pint Quantity when units='strict'")
    return value


def quantity_to_base(value: Any) -> Any:
    """Convert a pint quantity to the base units of the registry.

    Parameters
    ----------
    value : Any
        Quantity to convert.

    Returns
    -------
    Any
        The quantity expressed in base units.

    Raises
    ------
    SCDependencyError
        pint is not installed.
    """
    require_pint()
    return value.to_base_units()


def _dimensionless_magnitude(value: Any, fn_name: str) -> float:
    quantity = require_quantity(value, fn_name) if is_quantity(value) else value
    if is_quantity(quantity):
        return float(quantity.to(UNIT_REGISTRY.dimensionless).magnitude)
    return float(quantity)


def _exp(value: Any) -> Any:
    return np.exp(_dimensionless_magnitude(value, "exp")) * UNIT_REGISTRY.dimensionless


def _log(value: Any) -> Any:
    return np.log(_dimensionless_magnitude(value, "log")) * UNIT_REGISTRY.dimensionless


def _sin(value: Any) -> Any:
    return np.sin(_dimensionless_magnitude(value, "sin")) * UNIT_REGISTRY.dimensionless


def _cos(value: Any) -> Any:
    return np.cos(_dimensionless_magnitude(value, "cos")) * UNIT_REGISTRY.dimensionless


def _tanh(value: Any) -> Any:
    return np.tanh(_dimensionless_magnitude(value, "tanh")) * UNIT_REGISTRY.dimensionless


def _sinh(value: Any) -> Any:
    return np.sinh(_dimensionless_magnitude(value, "sinh")) * UNIT_REGISTRY.dimensionless


def _cosh(value: Any) -> Any:
    return np.cosh(_dimensionless_magnitude(value, "cosh")) * UNIT_REGISTRY.dimensionless


def _sigmoid(value: Any) -> Any:
    magnitude = _dimensionless_magnitude(value, "sigmoid")
    clipped = np.clip(magnitude, -500.0, 500.0)
    return (1.0 / (1.0 + np.exp(-clipped))) * UNIT_REGISTRY.dimensionless


def _sqrt(value: Any) -> Any:
    quantity = require_quantity(value, "sqrt") if is_quantity(value) else value
    if is_quantity(quantity):
        return quantity**0.5
    return np.sqrt(float(quantity)) * UNIT_REGISTRY.dimensionless


def _clip(value: Any, low: Any, high: Any) -> Any:
    quantity = require_quantity(value, "clip") if is_quantity(value) else value
    if is_quantity(quantity):
        low_quantity = require_quantity(low, "clip")
        high_quantity = require_quantity(high, "clip")
        low_mag = low_quantity.to(quantity.units).magnitude
        high_mag = high_quantity.to(quantity.units).magnitude
        clipped = np.clip(quantity.magnitude, low_mag, high_mag)
        return clipped * quantity.units
    return np.clip(float(quantity), float(low), float(high)) * UNIT_REGISTRY.dimensionless


def _round(value: Any) -> Any:
    """Round the base-unit magnitude, keeping the physical dimension and even ties."""
    if is_quantity(value):
        base = quantity_to_base(value)
        return np.rint(base.magnitude) * base.units
    return np.rint(float(value)) * UNIT_REGISTRY.dimensionless


def _heaviside(value: Any, at_zero: Any) -> Any:
    """Compare a base-unit magnitude with zero and require a dimensionless zero value."""
    magnitude = quantity_to_base(value).magnitude if is_quantity(value) else float(value)
    zero_value = _dimensionless_magnitude(at_zero, "heaviside at_zero")
    return np.heaviside(magnitude, zero_value) * UNIT_REGISTRY.dimensionless


def build_quantity_namespace() -> dict[str, Any]:
    """Return the unit-aware functions an expression may call in strict-units mode.

    Returns
    -------
    dict
        The names of the numerical equation namespace, bound to versions
        that require dimensionless arguments where the mathematics does.

    Raises
    ------
    SCDependencyError
        pint is not installed.
    """
    require_pint()
    return {
        "exp": _exp,
        "log": _log,
        "sqrt": _sqrt,
        "abs": abs,
        "sin": _sin,
        "cos": _cos,
        "tanh": _tanh,
        "cosh": _cosh,
        "sinh": _sinh,
        "sigmoid": _sigmoid,
        "pi": np.pi * UNIT_REGISTRY.dimensionless,
        "clip": _clip,
        "max": max,
        "min": min,
        "round": _round,
        "heaviside": _heaviside,
    }


def validate_quantity_expression(
    expr: str,
    env: dict[str, Any],
    *,
    expected_quantity: Any | None = None,
    label: str,
) -> Any:
    """Run one expression with unit-carrying values and check the unit of its result.

    Parameters
    ----------
    expr : str
        Authored expression. It passes the equation safety gate before it runs.
    env : dict
        Unit-carrying bindings and the functions of the quantity namespace.
    expected_quantity : Any, optional
        Quantity whose unit the result must be convertible to; no unit check
        when omitted.
    label : str
        Name of the expression used in refusals.

    Returns
    -------
    Any
        The value of the expression.

    Raises
    ------
    SCDependencyError
        pint is not installed.
    EquationRefusal
        The safety gate refuses the expression.
    ValueError
        A symbol is unknown or the expression cannot be evaluated.
    DimensionalError
        The units of the expression are inconsistent or differ from the
        expected unit.
    """
    require_pint()
    # The strict-units check runs in EquationNeuron's constructor before the
    # neuron's own validator is built. The translator takes the tree from the
    # safety gate itself, so the expression is checked here as well, and it
    # is run as a program of plain calls, never by the interpreter.
    program = compile_expression(expr)
    try:
        result = program(env)
    except NameError as exc:
        raise ValueError(f"Unknown symbol in {label}: {exc}") from exc
    except DimensionalError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not evaluate {label!r} under strict units") from exc

    if expected_quantity is None:
        return result

    if not is_quantity(result):
        raise DimensionalError(result, expected_quantity.units)

    result.to(expected_quantity.units)
    return result
