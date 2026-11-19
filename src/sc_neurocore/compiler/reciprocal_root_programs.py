# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded reciprocal square root integer datapath
"""Use bounded exact quotients to round reciprocal roots without overflowing signed64."""

from __future__ import annotations

from .function_unit_ir import Program
from .magnitude_function_programs import build_bounded_isqrt_with_remainder


def _power_of_two_quotient(
    p: Program,
    divisor: str,
    divisor_bits: int,
    exponent: int,
    quotient_bits: int,
    *,
    use_borrow: bool,
) -> tuple[str, str]:
    """Divide the leading power of two and retain a bounded exact remainder."""
    first = min(exponent, quotient_bits - 1)
    remainder = p.const(1 << (exponent - first))
    quotient = p.const(0)
    for position in range(first, -1, -1):
        shifted = remainder if position == first else p.shl(remainder, 1)
        if use_borrow:
            # The trial difference is in [-divisor, divisor), so this bit is its sign.
            difference = p.sub(shifted, divisor)
            take = p.sub(p.const(1), p.mask(p.shr(difference, divisor_bits), 1))
        else:
            take = p.le(divisor, shifted)
            difference = p.sub(shifted, divisor)
        bits = min(divisor_bits, exponent - position + 1)
        remainder = p.mask(p.select(take, difference, shifted), bits)
        quotient = p.add(p.shl(quotient, 1), take)
    return quotient, remainder


def build_rsqrt(p: Program, code: str, data_width: int, fraction: int) -> str:
    """Append the saturated, half-up reciprocal root to a signed64 program.

    Parameters
    ----------
    p : Program
        Destination evaluated and emitted by the maintained compiler targets.
    code : str
        Signed input identifier, narrowed to the admitted word before this call.
    data_width, fraction : int
        Common input and output format, validated by the public program factory.

    Returns
    -------
    str
        Result identifier. A bounded quotient followed by integer square root
        finds the greatest unsaturated root. The quotient and remainder give
        the exact rounding decision. All eagerly evaluated intermediates fit
        signed64 even for 32-bit inputs and 30 fractional bits. Nonpositive
        inputs return zero and cannot overflow a branch.
    """
    zero = p.const(0)
    positive = p.lt(zero, code)
    rail = (1 << (data_width - 1)) - 1
    maximum = p.const(rail)
    exponent = 3 * fraction
    quotient_bits = 2 * (data_width - 1)
    # The skipped prefix is smaller than every divisor that can use this quotient.
    overflow_limit = (1 << exponent) >> quotient_bits
    if overflow_limit >= rail:
        return p.select(positive, maximum, zero)
    overflow = p.le(code, p.const(overflow_limit))
    safe = p.select(overflow, p.const(overflow_limit + 1), code)
    # Reuse the subtraction as a comparison when wide fractions or a large
    # skipped prefix would otherwise require long comparator chains.
    use_borrow = overflow_limit >= 1 << 8 or (data_width > 18 and fraction >= 16)
    quotient, remainder = _power_of_two_quotient(
        p, safe, data_width - 1, exponent, quotient_bits, use_borrow=use_borrow
    )
    top_pair = 2 * min(data_width - 2, 3 * fraction // 2)
    root, square_remainder = build_bounded_isqrt_with_remainder(
        p, quotient, top_pair, use_borrow=use_borrow
    )
    # The squared midpoint is r*r + r + 1/4, and the square root retains q - r*r.
    at_midpoint = p.eq(square_remainder, root)
    fraction_reaches_midpoint = p.le(safe, p.shl(remainder, 2))
    round_up = p.select(at_midpoint, fraction_reaches_midpoint, p.lt(root, square_remainder))
    rounded = p.add(root, round_up)
    saturated = p.select(p.lt(maximum, rounded), maximum, rounded)
    return p.select(positive, p.select(overflow, maximum, saturated), zero)
