# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Validated conversion of caller input to one byte per bit

"""One validated conversion of caller input to a bit array.

A bit array is a non-empty boolean, integer or real array in which every
element equals 0 or 1 *as given*. The check reads the original values;
conversion to one byte per bit happens only afterwards, so a value such as
``0.9``, ``256``, ``-1``, ``NaN`` or infinity is refused and is never turned
into a bit by the conversion. Text, bytes, objects and complex numbers are
refused by their element type.

The shape rule belongs to the caller: the cross-correlation diagnostics need
a one-dimensional stream, the bipolar primitives take any shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]

#: NumPy element kinds that may hold bits: boolean, signed and unsigned integer, real.
_BIT_CONTAINER_KINDS = frozenset("biuf")


def as_bit_array(bits: Any, name: str, *, one_dimensional: bool = False) -> UInt8Array:
    """Return ``bits`` as one byte per bit after checking the original values.

    The element type, the dimension, the size and every value are checked on
    the array as NumPy reads it from ``bits``, before any conversion, so no
    value becomes 0 or 1 through a cast.

    Parameters
    ----------
    bits : array-like
        Candidate bits in any boolean, integer or real container.
    name : str
        Name of the caller's argument, used in every refusal.
    one_dimensional : bool, default False
        Require exactly one dimension. When false, any shape is admitted.

    Returns
    -------
    numpy.ndarray of uint8
        A new array of the same shape holding 0 and 1. The input is never
        written to and never returned itself.

    Raises
    ------
    ValueError
        If the input is a nested sequence of unequal lengths, has an element
        type other than boolean, integer or real, breaks the shape rule, is
        empty, or holds a value other than 0 and 1.
    """
    shape_rule = "one-dimensional" if one_dimensional else "a rectangular array"
    try:
        original = np.asarray(bits)
    except ValueError as error:
        # NumPy refuses nested sequences of unequal length.
        raise ValueError(f"{name} must be {shape_rule}") from error
    if original.dtype.kind not in _BIT_CONTAINER_KINDS:
        raise ValueError(f"{name} must be a boolean, integer or real array of 0/1 bits")
    if one_dimensional and original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if original.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all((original == 0) | (original == 1)):
        raise ValueError(f"{name} must contain only 0/1 bits")
    return original.astype(np.uint8)
