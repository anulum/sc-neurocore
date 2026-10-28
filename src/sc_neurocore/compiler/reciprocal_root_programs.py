# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded reciprocal square root integer datapath
"""Compare split products to round reciprocal roots without overflowing signed64."""

from __future__ import annotations

from .function_unit_ir import Program


def _product(p: Program, a: str, b: str) -> tuple[str, str]:
    """Split a positive 31-by-62-bit product into base-2**32 limbs."""
    low = p.mul(a, p.mask(b, 32))
    high = p.add(p.mul(a, p.shr(b, 32)), p.shr(low, 32))
    return high, p.mask(low, 32)


def _le(p: Program, high: str, low: str, limit_high: str, limit_low: str) -> str:
    """Compare nonnegative two-limb integers without combining their limbs."""
    return p.select(p.eq(high, limit_high), p.le(low, limit_low), p.lt(high, limit_high))


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
        Result identifier. A bitwise search finds the greatest unsaturated root
        with ``code * root**2 <= 2**(3*fraction)``. An exact midpoint comparison
        rounds it. All eagerly evaluated intermediates fit signed64 even for
        32-bit inputs and 30 fractional bits. Nonpositive inputs return zero,
        the documented out-of-domain sentinel, and cannot overflow a branch.
    """
    zero, one = p.const(0), p.const(1)
    positive = p.lt(zero, code)
    safe = p.select(positive, code, one)
    numerator = 1 << (3 * fraction)
    nh, nl = p.const(numerator >> 32), p.const(numerator & 0xFFFFFFFF)
    root = zero
    for position in range(data_width - 2, -1, -1):
        trial = p.add(root, p.const(1 << position))
        high, low = _product(p, safe, p.mul(trial, trial))
        root = p.select(_le(p, high, low, nh, nl), trial, root)
    high, low = _product(p, safe, p.add(p.mul(root, root), root))
    low = p.add(low, p.shr(safe, 2))
    high = p.add(high, p.shr(low, 32))
    low = p.mask(low, 32)
    # N >= a*(r+1/2)**2 iff N >= ceil(a*(r*r+r) + a/4).
    remainder = p.lt(zero, p.mask(safe, 2))
    low = p.add(low, remainder)
    high = p.add(high, p.shr(low, 32))
    low = p.mask(low, 32)
    rounded = p.add(root, _le(p, high, low, nh, nl))
    maximum = p.const((1 << (data_width - 1)) - 1)
    saturated = p.select(p.lt(maximum, rounded), maximum, rounded)
    return p.select(positive, saturated, zero)
