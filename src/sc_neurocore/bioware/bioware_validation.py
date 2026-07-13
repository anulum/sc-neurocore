# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-closed biological-interface validation

"""Fail-closed validation shared by biological-interface boundaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def require_finite(value: float, name: str) -> None:
    """Require a finite scalar value.

    Parameters
    ----------
    value:
        Scalar to validate.
    name:
        Field name used in the error message.

    Raises
    ------
    TypeError
        If ``value`` is a boolean or not a real scalar.
    ValueError
        If ``value`` is NaN or infinite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def require_nonnegative(value: float, name: str) -> None:
    """Require a finite scalar greater than or equal to zero."""
    require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def require_positive(value: float, name: str) -> None:
    """Require a finite scalar strictly greater than zero."""
    require_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")


def require_nonnegative_int(value: int, name: str) -> None:
    """Require a non-boolean integer greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def require_positive_int(value: int, name: str) -> None:
    """Require a non-boolean integer strictly greater than zero."""
    require_nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be > 0")


def validate_voltage_matrix(
    voltage_data: np.ndarray[Any, Any],
    *,
    expected_channels: int | None = None,
) -> None:
    """Validate a non-empty, finite two-dimensional MEA voltage matrix.

    Parameters
    ----------
    voltage_data:
        Matrix with shape ``(samples, channels)``.
    expected_channels:
        Optional exact channel count.

    Raises
    ------
    TypeError
        If the input is not a NumPy array with a numeric dtype.
    ValueError
        If dimensionality, shape, channel count, or finiteness is invalid.
    """
    if not isinstance(voltage_data, np.ndarray):
        raise TypeError("voltage_data must be a NumPy array")
    if not np.issubdtype(voltage_data.dtype, np.number):
        raise TypeError("voltage_data must have a numeric dtype")
    if voltage_data.ndim != 2:
        raise ValueError("voltage_data must have shape (samples, channels)")
    if voltage_data.shape[0] == 0 or voltage_data.shape[1] == 0:
        raise ValueError("voltage_data must contain at least one sample and channel")
    if expected_channels is not None and voltage_data.shape[1] != expected_channels:
        raise ValueError(
            f"voltage_data has {voltage_data.shape[1]} channels; expected {expected_channels}"
        )
    if not np.all(np.isfinite(voltage_data)):
        raise ValueError("voltage_data must contain only finite values")


def validate_binary_bitstream(
    bitstream: np.ndarray[Any, Any],
    *,
    name: str,
    allow_empty: bool,
) -> None:
    """Validate a one-dimensional NumPy bitstream containing only 0 and 1."""
    if not isinstance(bitstream, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if bitstream.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if bitstream.size == 0:
        if allow_empty:
            return
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(bitstream.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(bitstream)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((bitstream == 0) | (bitstream == 1)):
        raise ValueError(f"{name} must contain only binary values 0 and 1")
