# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded signed64 programs for real powers
"""Build Q56 powers with split products and restoring fractional division.

Every operation, including an unselected branch, fits signed64. Products are
formed as three base-2**31 limbs; their high limb can contain up to 60 bits.
The reference in ``power_units`` instead uses unbounded Python products.
"""

from __future__ import annotations

from functools import lru_cache

from .function_unit_ir import Program
from .function_units import check_format
from .power_units import INTERNAL_FRACTION, PowerExponent, coefficients, exponent_format

_SCALE = 1 << INTERNAL_FRACTION


def _absolute(p: Program, value: str) -> str:
    """Take an absolute value whose admitted inputs exclude the signed64 minimum."""
    return p.select(p.lt(value, p.const(0)), p.neg(value), value)


def _minimum(p: Program, value: str, maximum: str) -> str:
    """Cap a nonnegative value before a subsequent narrowing operation."""
    return p.select(p.lt(maximum, value), maximum, value)


def _scaled_product(p: Program, a: str, b: str, shift: int) -> str:
    """Multiply magnitudes below 2**61, right-shift 0..56 bits and cap at 2**62."""
    low = p.mul(p.mask(a, 31), p.mask(b, 31))
    middle = p.add(
        p.add(p.mul(p.shr(a, 31), p.mask(b, 31)), p.mul(p.mask(a, 31), p.shr(b, 31))),
        p.shr(low, 31),
    )
    high = p.add(p.mul(p.shr(a, 31), p.shr(b, 31)), p.shr(middle, 31))
    overflow = p.le(p.const(1 << shift), high)
    safe_high = p.select(overflow, p.const(0), high)
    result = p.shl(safe_high, 62 - shift)
    if shift <= 31:
        result = p.add(result, p.shl(p.mask(middle, 31), 31 - shift))
        result = p.add(result, p.shr(p.mask(low, 31), shift))
    else:
        result = p.add(result, p.shr(p.mask(middle, 31), shift - 31))
    return p.select(overflow, p.const(1 << 62), result)


def _leading_bit(p: Program, magnitude: str) -> str:
    """Find the leading-bit position of a nonzero magnitude at most 2**31."""
    rest, lead = magnitude, p.const(0)
    for width in (16, 8, 4, 2, 1):
        take = p.le(p.const(1 << width), rest)
        rest = p.select(take, p.shr(rest, width), rest)
        lead = p.select(take, p.add(lead, p.const(width)), lead)
    return lead


def _ratio(p: Program, numerator: str, denominator: str) -> str:
    """Form floor(2**56*numerator/denominator) for 0<=numerator<denominator<2**33."""
    remainder, quotient = numerator, p.const(0)
    for _ in range(INTERNAL_FRACTION):
        remainder = p.shl(remainder, 1)
        take = p.le(denominator, remainder)
        remainder = p.select(take, p.sub(remainder, denominator), remainder)
        quotient = p.add(p.shl(quotient, 1), take)
    return quotient


def _logarithm(p: Program, magnitude: str, fraction: int) -> str:
    """Compute a normalized logarithm without clipping it to the output format."""
    lead = _leading_bit(p, magnitude)
    leading = p.shl_var(p.const(1), lead)
    t = _ratio(p, p.sub(magnitude, leading), p.add(magnitude, leading))
    squared = _scaled_product(p, t, t, INTERNAL_FRACTION)
    log_coefficients, _ = coefficients()
    value = p.const(log_coefficients[-1])
    for coefficient in reversed(log_coefficients[:-1]):
        value = p.add(p.const(coefficient), _scaled_product(p, value, squared, INTERNAL_FRACTION))
    mantissa = _scaled_product(p, t, value, INTERNAL_FRACTION)
    return p.add(p.shl(p.sub(lead, p.const(fraction)), INTERNAL_FRACTION), mantissa)


def _exponential(p: Program, exponent: str) -> tuple[str, str]:
    """Reduce a signed Q56 exponent in [-64,64] and evaluate its fractional exp2."""
    whole = p.shr(exponent, INTERNAL_FRACTION)
    rest = p.sub(exponent, p.shl(whole, INTERNAL_FRACTION))
    _, exp_coefficients = coefficients()
    mantissa = p.const(exp_coefficients[-1])
    for coefficient in reversed(exp_coefficients[:-1]):
        mantissa = p.add(
            p.const(coefficient), _scaled_product(p, mantissa, rest, INTERNAL_FRACTION)
        )
    return whole, mantissa


def _round_output(p: Program, whole: str, mantissa: str, data_width: int, fraction: int) -> str:
    """Round once, capping magnitude before applying the result's sign."""
    cap = p.const(1 << (data_width - 1))
    shift = p.add(whole, p.const(fraction - INTERNAL_FRACTION))
    overflow = p.le(p.const(data_width - 1 - INTERNAL_FRACTION), shift)
    right = _minimum(p, p.neg(shift), p.const(62))
    right = p.select(p.lt(right, p.const(1)), p.const(1), right)
    half = p.shl_var(p.const(1), p.sub(right, p.const(1)))
    rounded = p.shr_var(p.add(mantissa, half), right)
    return p.select(overflow, cap, _minimum(p, rounded, cap))


@lru_cache(maxsize=128)
def _program(data_width: int, fraction: int, literal: PowerExponent | None) -> Program:
    """Build the validated constant or indexed-input datapath once per specialization."""
    p = Program()
    base, zero = p.input(), p.const(0)
    if literal is not None and literal.value == 0:
        p.output = p.const(min(1 << fraction, (1 << (data_width - 1)) - 1))
        return p
    exponent = p.input(1) if literal is None else p.const(literal.code)
    ef = fraction if literal is None else literal.fraction
    magnitude = _absolute(p, base)
    nonzero = p.lt(zero, magnitude)
    safe = p.select(nonzero, magnitude, p.const(1))
    logarithm = _logarithm(p, safe, fraction)
    product = _scaled_product(p, _absolute(p, logarithm), _absolute(p, exponent), ef)
    negative_exponent = p.xor(p.lt(logarithm, zero), p.lt(exponent, zero))
    product = p.select(negative_exponent, p.neg(product), product)
    whole, mantissa = _exponential(p, product)
    rounded = _round_output(p, whole, mantissa, data_width, fraction)
    positive = _minimum(p, rounded, p.const((1 << (data_width - 1)) - 1))
    signed = (
        p.select(p.lt(base, zero), p.neg(rounded), positive)
        if literal is not None and literal.odd
        else positive
    )
    valid = nonzero if literal is not None and literal.integral else p.lt(zero, base)
    p.output = p.select(valid, signed, zero)
    return p


def power_program(data_width: int, fraction: int, exponent: int | float | None = None) -> Program:
    """Return one v2 power unit using only existing signed64 program operations.

    Parameters
    ----------
    data_width, fraction : int
        Signed base, dynamic-exponent and output Q format. Inputs to the program
        must already be narrowed to this word, as the maintained emitters do.
    exponent : int or float or None, optional
        Finite constant specialization with independently encoded precision.
        None creates a two-input unit for a dynamic signed Q-word exponent.

    Returns
    -------
    Program
        Saturated power datapath, bit-identical to ``power_code``. Its positive
        domain and literal negative-base/zero policies are defined there. No
        table, new native opcode or wide-integer ABI is required.

    Raises
    ------
    ValueError
        If the format or exponent is invalid. Validation precedes memoization,
        including boolean/literal aliases of a previously cached specialization.
    """
    if type(data_width) is not int or type(fraction) is not int:
        raise ValueError("power format widths must be integers")
    check_format(data_width, fraction)
    literal = None if exponent is None else exponent_format(exponent)
    return _program(data_width, fraction, literal)
