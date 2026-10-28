# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point Euclidean magnitude
"""Evaluate Euclidean magnitude with one rounding in the declared output word."""

from __future__ import annotations

from math import isqrt


def hypot_code(first: int, second: int, data_width: int, fraction: int) -> int:
    """Return the saturated, nearest code of a two-coordinate Euclidean norm.

    Parameters
    ----------
    first, second : int
        Signed codes admitted by the public function-unit dispatcher.
    data_width : int
        Width of the signed input and result words.
    fraction : int
        Shared fractional width; the common scale cancels in code space.

    Returns
    -------
    int
        Nearest integer to ``sqrt(first**2 + second**2)``, capped at the
        positive signed-word maximum. An exact half-integer square root of
        an integer is impossible, so the rounding rule has no tie case.
    """
    squared = first * first + second * second
    root = isqrt(squared)
    rounded = root + int(squared - root * root > root)
    return min(rounded, (1 << (data_width - 1)) - 1)
