# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Euclidean magnitude integer datapath
"""Build a norm datapath whose intermediates stay within signed 64-bit words."""

from __future__ import annotations

from .function_unit_ir import Program


def build_hypot(p: Program, first: str, second: str, data_width: int, fraction: int) -> str:
    """Append an exactly rounded two-coordinate magnitude to an integer program.

    Parameters
    ----------
    p : Program
        Destination signed-64-bit program, evaluated by every maintained target.
    first, second : str
        Input identifiers carrying signed codes of the admitted word format.
    data_width : int
        Input and output width, at most 32 bits.
    fraction : int
        Common fractional width; no rescaling is needed before taking the norm.

    Returns
    -------
    str
        Result identifier, rounded to nearest and saturated at the positive
        signed-word maximum. Clipping each absolute input at that maximum
        preserves saturation and keeps the sum of squares within signed 64 bits.
    """
    zero = p.const(0)
    high = p.const((1 << (data_width - 1)) - 1)
    a = p.select(p.lt(first, zero), p.neg(first), first)
    b = p.select(p.lt(second, zero), p.neg(second), second)
    a = p.select(p.lt(high, a), high, a)
    b = p.select(p.lt(high, b), high, b)
    squared = p.add(p.mul(a, a), p.mul(b, b))
    root, remainder = zero, squared
    for position in range(2 * data_width - 2, -1, -2):
        bit = p.const(1 << position)
        trial = p.add(root, bit)
        take = p.le(trial, remainder)
        remainder = p.select(take, p.sub(remainder, trial), remainder)
        root = p.select(take, p.add(p.shr(root, 1), bit), p.shr(root, 1))
    rounded = p.add(root, p.lt(root, remainder))
    return p.select(p.lt(high, rounded), high, rounded)
