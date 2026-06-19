# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional pint helpers for strict dimensional checking

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.exceptions import SCDependencyError

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
    if not HAS_PINT:
        raise SCDependencyError(
            "pint is not available. Install with: pip install sc-neurocore[units]"
        )


def is_quantity(value: Any) -> bool:
    return HAS_PINT and isinstance(value, pint.Quantity)


def require_quantity(value: Any, label: str) -> Any:
    require_pint()
    if not is_quantity(value):
        raise ValueError(f"{label} must be a pint Quantity when units='strict'")
    return value


def quantity_to_base(value: Any) -> Any:
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


def build_quantity_namespace() -> dict[str, Any]:
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
    }


def validate_quantity_expression(
    expr: str,
    env: dict[str, Any],
    *,
    expected_quantity: Any | None = None,
    label: str,
) -> Any:
    require_pint()
    code = compile(expr, f"<units:{label}>", "eval")
    try:
        result = eval(code, {"__builtins__": {}}, env)  # nosec B307
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
