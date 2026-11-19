# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded nonnegative integer datapaths
"""Build magnitude arithmetic whose intermediates fit signed 64-bit words."""

from __future__ import annotations

from .function_unit_ir import Program


def build_fractional_ratio(p: Program, low: str, high: str, width: int, fraction: int) -> str:
    """Append an exact truncated fraction for two ordered nonnegative magnitudes.

    Parameters
    ----------
    p : Program
        Destination signed-64-bit arithmetic program.
    low, high : str
        Integer value identifiers with zero <= low <= high and high > 0.
    width : int
        Number of bits that represent the divisor.
    fraction : int
        Number of fractional quotient bits to emit.

    Returns
    -------
    str
        Identifier for the truncated quotient. The remainder stays strictly
        below the divisor after each subtraction. Equality produces exactly one.

    """
    equal = p.eq(low, high)
    zero = p.const(0)
    remainder = p.select(equal, zero, low)
    quotient = zero
    for remaining_bits in range(fraction - 1, -1, -1):
        shifted = p.shl(remainder, 1)
        take = p.le(high, shifted)
        if remaining_bits:
            remainder = p.mask(p.select(take, p.sub(shifted, high), shifted), width)
        quotient = p.add(p.shl(quotient, 1), take)
    return p.select(equal, p.const(1 << fraction), quotient)


def build_bounded_isqrt(p: Program, radicand: str, top_pair: int) -> str:
    """Append the exact integer square root of a bounded nonnegative radicand.

    Parameters
    ----------
    p : Program
        Destination evaluated by the maintained compiler targets.
    radicand : str
        Identifier for a nonnegative integer with no bits above the first pair.
    top_pair : int
        Even position of the low bit of that first pair.

    Returns
    -------
    str
        Identifier for the floor of the square root. After each pair, the
        consumed prefix equals the square of the root plus the remainder.
        The remainder is at most twice the root, so each mask is exact.

    """
    return _append_bounded_isqrt(p, radicand, top_pair, retain_remainder=False)[0]


def build_bounded_isqrt_with_remainder(
    p: Program, radicand: str, top_pair: int, *, use_borrow: bool = False
) -> tuple[str, str]:
    """Append the floor square root and its exact nonnegative squared remainder.

    The borrow form uses the sign of the same bounded difference that supplies
    the remainder when the trial root fits.
    """
    return _append_bounded_isqrt(
        p, radicand, top_pair, retain_remainder=True, use_borrow=use_borrow
    )


def _append_bounded_isqrt(
    p: Program, radicand: str, top_pair: int, *, retain_remainder: bool, use_borrow: bool = False
) -> tuple[str, str]:
    """Carry the remaining radicand through one two-bit step at a time."""
    root = p.const(0)
    remainder = root if use_borrow else p.const(0)
    one = p.const(1)
    for root_bits, position in enumerate(range(top_pair, -1, -2), start=1):
        digit = p.mask(p.shr(radicand, position), 2)
        shifted = p.add(p.shl(remainder, 2), digit)
        trial = p.add(p.shl(root, 2), one)
        if use_borrow:
            difference = p.sub(shifted, trial)
            # At this stage both operands fit in root_bits + 2 unsigned bits.
            take = p.sub(one, p.mask(p.shr(difference, root_bits + 2), 1))
        else:
            difference = None
            take = p.le(trial, shifted)
        if position or retain_remainder:
            if difference is None:
                difference = p.sub(shifted, trial)
            remainder = p.mask(p.select(take, difference, shifted), root_bits + 1)
        root = p.mask(p.add(p.shl(root, 1), take), root_bits)
    return root, remainder


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
