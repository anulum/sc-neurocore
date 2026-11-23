# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing input-scalar validation

"""Validate the user-supplied scalars of the quantum-annealing bridge.

This module is the one owner of the annealing input checks. A real-valued field
accepts any :class:`numbers.Real` except a boolean: Python and NumPy integers and
floats and :class:`fractions.Fraction`. It refuses ``bool`` and ``numpy.bool_``,
strings, :class:`decimal.Decimal` and arrays, then requires a finite value. An
integer field accepts any :class:`numbers.Integral` except a boolean. Each
refusal is a :class:`ValueError` naming the field.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = ["finite_real", "non_negative_index", "positive_int", "positive_real"]


def _is_boolean(value: object) -> bool:
    """Return whether a value is a Python or NumPy boolean."""
    return isinstance(value, (bool, np.bool_))


def finite_real(name: str, value: object) -> float:
    """Return a real, non-boolean, finite scalar as a float.

    Parameters
    ----------
    name : str
        Field name used in the refusal message.
    value : object
        Candidate scalar.

    Returns
    -------
    float
        The value converted to a finite float.

    Raises
    ------
    ValueError
        "``name`` must be a real number" for a boolean or a non-real value, and
        "``name`` must be finite" for a value whose float is infinite or NaN,
        including an integer too large for a float.
    """
    if _is_boolean(value) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def positive_real(name: str, value: object) -> float:
    """Return a finite real scalar that is greater than zero.

    Parameters
    ----------
    name : str
        Field name used in the refusal message.
    value : object
        Candidate scalar.

    Returns
    -------
    float
        The value converted to a positive finite float.

    Raises
    ------
    ValueError
        The refusals of :func:`finite_real`, and "``name`` must be greater than
        zero" for zero or a negative value.
    """
    numeric = finite_real(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return numeric


def positive_int(name: str, value: object) -> int:
    """Return a non-boolean integer that is greater than zero.

    Parameters
    ----------
    name : str
        Field name used in the refusal message.
    value : object
        Candidate integer, Python or NumPy.

    Returns
    -------
    int
        The value as a Python integer.

    Raises
    ------
    ValueError
        "``name`` must be a positive integer" for a boolean, a non-integer, zero
        or a negative value.
    """
    if _is_boolean(value) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def non_negative_index(name: str, value: object) -> int:
    """Return a non-boolean integer that is zero or greater.

    Parameters
    ----------
    name : str
        Field name used in the refusal message.
    value : object
        Candidate index, Python or NumPy.

    Returns
    -------
    int
        The value as a Python integer.

    Raises
    ------
    ValueError
        "``name`` must be a non-negative integer" for a boolean, a non-integer or
        a negative value.
    """
    if _is_boolean(value) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
