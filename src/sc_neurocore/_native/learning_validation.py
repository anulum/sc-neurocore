# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native-learning FFI validation contracts

"""Fail-closed scalar and array validation for native learning backends."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

RULE_ELIGENT = 0
RULE_STDP = 1
RULE_REWARD_STDP = 2
RULE_BCM = 3
VALID_RULE_TYPES = frozenset({RULE_ELIGENT, RULE_STDP, RULE_REWARD_STDP, RULE_BCM})

MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
MIN_I32 = -(1 << 31)
MAX_I32 = (1 << 31) - 1
MAX_ONLINE_O1_WEIGHT_BITS = 31
MAX_ONLINE_O1_TRACE_BITS = 30
MAX_ONLINE_O1_REWARD_BITS = 30
MAX_ONLINE_O1_SHIFT = 30


def require_integral(*, name: str, value: object) -> int:
    """Return an integer after rejecting booleans and non-integral values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and must not be bool")
    return int(value)


def require_non_negative_integral(*, name: str, value: object) -> int:
    """Return a non-negative integer for an unsigned native ABI field."""
    integral = require_integral(name=name, value=value)
    if integral < 0:
        raise ValueError(f"{name} must be >= 0")
    return integral


def require_integral_range(
    *,
    name: str,
    value: object,
    lower: int,
    upper: int,
) -> int:
    """Return an integer inside the inclusive native ABI domain."""
    integral = require_integral(name=name, value=value)
    if integral < lower or integral > upper:
        raise ValueError(f"{name} must be in {lower}..={upper}")
    return integral


def require_count(value: object) -> int:
    """Return a positive layer size that can safely cross ``size_t``."""
    count = require_integral(name="count", value=value)
    if count <= 0:
        raise ValueError("count must be > 0")
    if count > MAX_U32:
        raise ValueError(f"count must be <= {MAX_U32}")
    return count


def require_rule_type(value: object) -> int:
    """Return one of the four native plasticity rule identifiers."""
    rule_type = require_integral(name="rule_type", value=value)
    if rule_type not in VALID_RULE_TYPES:
        expected = ", ".join(str(item) for item in sorted(VALID_RULE_TYPES))
        raise ValueError(f"rule_type must be one of {expected}")
    return rule_type


def require_bool(*, name: str, value: object) -> bool:
    """Return a real Python boolean for the C ``bool`` ABI."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def require_finite_float(*, name: str, value: object) -> float:
    """Return a finite real scalar after rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and must not be bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_positive_float(*, name: str, value: object) -> float:
    """Return a finite strictly positive scalar."""
    result = require_finite_float(name=name, value=value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def require_non_negative_float(*, name: str, value: object) -> float:
    """Return a finite non-negative scalar."""
    result = require_finite_float(name=name, value=value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def require_unit_interval(*, name: str, value: object) -> float:
    """Return a finite scalar in the closed unit interval."""
    result = require_finite_float(name=name, value=value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def require_u32_seed(*, name: str, value: object) -> int:
    """Return a deterministic seed accepted by CPU and WGPU paths."""
    return require_integral_range(name=name, value=value, lower=0, upper=MAX_U32)


def require_u64_seed(*, name: str, value: object) -> int:
    """Return an explicit seed accepted by the Rayon analogue path."""
    return require_integral_range(name=name, value=value, lower=0, upper=MAX_U64)


def saturate(value: int, lower: int, upper: int) -> int:
    """Clamp an already validated integer into inclusive bounds."""
    return min(upper, max(lower, value))


def _require_vector(array: np.ndarray[Any, Any], *, name: str, length: int | None) -> None:
    """Validate one-dimensional shape and optional native layer length."""
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}")


def as_bool_vector(
    values: object,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray[Any, Any]:
    """Return a contiguous Boolean vector without truthiness coercion."""
    raw = np.asarray(values)
    _require_vector(raw, name=name, length=length)
    if raw.dtype != np.bool_:
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain only booleans or binary numbers") from exc
        if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
            raise ValueError(f"{name} must contain only boolean, 0, or 1 values")
    return np.ascontiguousarray(raw, dtype=np.bool_)


def as_float_vector(
    values: object,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray[Any, Any]:
    """Return a contiguous finite ``float32`` vector."""
    raw = np.asarray(values)
    _require_vector(raw, name=name, length=length)
    try:
        result = np.ascontiguousarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def as_probability_vector(
    values: object,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray[Any, Any]:
    """Return a contiguous finite probability vector in ``[0, 1]``."""
    result = as_float_vector(values, name=name, length=length)
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} must contain only probabilities in [0, 1]")
    return result
