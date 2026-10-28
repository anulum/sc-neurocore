# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High precision fixed-point power reference
"""Specify power arithmetic independently of the bounded emitted integer program.

The positive-base path computes ``exp2(y * log2(x))`` with 56 fractional bits.
Logarithms use the atanh series on a normalized mantissa; exponentials use a
degree-20 Taylor polynomial. No intermediate is narrowed to the output word.
Literal exponents have their own format, independent of the base's Q format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from functools import cache

from .function_units import check_format

INTERNAL_FRACTION = 56
_SCALE = 1 << INTERNAL_FRACTION


@dataclass(frozen=True, slots=True)
class PowerExponent:
    """A literal exponent with independent precision and original domain flags.

    Parameters
    ----------
    value : int or float
        Exact original finite value, canonicalized to int when integral.
        Zero/sign/integrality are tested before coding.
    code, fraction : int
        Signed exponent code and fractional width, with magnitude below 2**61.
        Magnitudes above 2**36 are capped: every nonunit admitted base then
        already forces the output to zero or saturation.
    integral, odd : bool
        Exact original integrality and odd parity, used for negative bases.
    """

    value: int | float
    code: int
    fraction: int
    integral: bool
    odd: bool


def exponent_format(value: int | float) -> PowerExponent:
    """Encode a finite literal without losing its original domain classification.

    Parameters
    ----------
    value : int or float
        Literal exponent; booleans are excluded.

    Returns
    -------
    PowerExponent
        Independent fixed-point code rounded to nearest, ties away from zero.

    Raises
    ------
    ValueError
        If the value is not a finite real Python numeric literal.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("power exponent must be a finite real numeric literal")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("power exponent must be finite")
    magnitude = min(abs(Fraction(value)), Fraction(1 << 36))
    bits = max(0, magnitude.numerator.bit_length() - magnitude.denominator.bit_length() + 1)
    fraction = min(INTERNAL_FRACTION, 61 - bits)
    numerator = magnitude.numerator << fraction
    code = (2 * numerator + magnitude.denominator) // (2 * magnitude.denominator)
    integral = isinstance(value, int) or value.is_integer()
    return PowerExponent(
        int(value) if integral else value,
        -code if value < 0 else code,
        fraction,
        integral,
        integral and bool(int(value) & 1),
    )


@cache
def coefficients() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return Q56 logarithm and exponential coefficients generated at precision 96.

    Returns
    -------
    tuple of tuple of int
        Twenty coefficients of ``(2/ln(2))*sum(t**(2*k)/(2*k+1))`` and
        twenty-one coefficients of ``exp(ln(2)*r)`` in increasing degree.
        Both reduced arguments are nonnegative: ``t < 1/3`` and ``r < 1``.
    """
    with localcontext() as context:
        context.prec = 96
        ln2 = Decimal(2).ln()
        log = tuple(
            int(
                (Decimal(2) * _SCALE / (ln2 * (2 * index + 1))).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )
            for index in range(20)
        )
        exp = tuple(
            int(
                (ln2**index * _SCALE / math.factorial(index)).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )
            for index in range(21)
        )
    return log, exp


def _word(code: int, data_width: int) -> None:
    """Refuse a Boolean, noninteger or code outside the admitted signed word."""
    if type(code) is not int or not -(1 << (data_width - 1)) <= code < 1 << (data_width - 1):
        raise ValueError("power input code must be a signed word of the configured width")


def power_code(
    base: int,
    data_width: int,
    fraction: int,
    *,
    exponent: int | float | None = None,
    exponent_code: int | None = None,
) -> int:
    """Evaluate a power reference using arbitrary-precision integer arithmetic.

    Parameters
    ----------
    base : int
        Signed base code of the configured format.
    data_width, fraction : int
        Common base, dynamic-exponent and output format.
    exponent : int or float or None, optional
        Finite literal exponent, encoded independently. None selects a dynamic
        exponent, supplied through ``exponent_code``.
    exponent_code : int or None, optional
        Dynamic signed Q-word exponent; required only in dynamic mode.

    Returns
    -------
    int
        Rounded Q56 approximation, with magnitude ties rounded up and signed
        word saturation. Dynamic powers require positive bases. Integral
        literals also admit negative bases; positive literals admit zero.
        Literal zero produces saturated one, including for a zero base.
        Inputs outside these mathematical domains return the zero sentinel.

    Raises
    ------
    ValueError
        For an invalid format, operand word, literal or exponent argument pair.
    """
    if type(data_width) is not int or type(fraction) is not int:
        raise ValueError("power format widths must be integers")
    check_format(data_width, fraction)
    _word(base, data_width)
    if exponent is None:
        if exponent_code is None:
            raise ValueError("dynamic power needs an exponent code")
        _word(exponent_code, data_width)
        code, ef, negative = exponent_code, fraction, False
        if base <= 0:
            return 0
    else:
        if exponent_code is not None:
            raise ValueError("literal power excludes an exponent code")
        literal = exponent_format(exponent)
        code, ef = literal.code, literal.fraction
        if literal.value == 0:
            return min(1 << fraction, (1 << (data_width - 1)) - 1)
        if base == 0 or base < 0 and not literal.integral:
            return 0
        negative = base < 0 and literal.odd
    magnitude = abs(base)
    lead = magnitude.bit_length() - 1
    leading = 1 << lead
    t = ((magnitude - leading) << INTERNAL_FRACTION) // (magnitude + leading)
    squared = t * t >> INTERNAL_FRACTION
    log_coefficients, exp_coefficients = coefficients()
    logarithm = log_coefficients[-1]
    for coefficient in reversed(log_coefficients[:-1]):
        logarithm = coefficient + (logarithm * squared >> INTERNAL_FRACTION)
    logarithm = ((lead - fraction) << INTERNAL_FRACTION) + (t * logarithm >> INTERNAL_FRACTION)
    product = min(abs(logarithm) * abs(code) >> ef, 64 * _SCALE)
    if (logarithm < 0) != (code < 0):
        product = -product
    whole, rest = divmod(product, _SCALE)
    mantissa = exp_coefficients[-1]
    for coefficient in reversed(exp_coefficients[:-1]):
        mantissa = coefficient + (mantissa * rest >> INTERNAL_FRACTION)
    shift = whole + fraction - INTERNAL_FRACTION
    rounded = mantissa << shift if shift >= 0 else (mantissa + (1 << (-shift - 1))) >> -shift
    signed = -rounded if negative else rounded
    return max(-(1 << (data_width - 1)), min((1 << (data_width - 1)) - 1, signed))
