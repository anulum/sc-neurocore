# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2: the integer reference of every function unit

"""Integer reference of the ``v2`` function units that the RTL and the bit-true kernel mirror.

``lut-v1`` reads one sample of a fixed 256-entry table (see
:mod:`sc_neurocore.compiler.lut_accuracy` for what that costs). ``v2`` computes
each function from a range reduction and a piecewise-linear table over the
reduced argument, interpolated quadratically within each segment, at an internal
fraction of :data:`P` bits, and rounds once to the output format:

* ``exp`` — ``x·log2(e) = k + f``; ``2^f`` from a table on ``[0, 1)``; shift by ``k``.
* ``log`` — the leading one of the input gives ``k``; ``log2`` of the mantissa from
  a table on ``[1, 2)``; ``(k + log2 m)·ln 2``.
* ``sqrt`` — exact integer square root, rounded to nearest (no table).
* ``sin``/``cos`` — the argument in turns, reduced modulo one turn by the word's
  wrap; a quarter-wave table and quadrant symmetry.
* ``sigmoid`` — a table on ``[0, 32)`` and ``σ(-x) = 1 - σ(x)``; ``tanh(x) = 2σ(2x) - 1``.
* ``cosh``/``sinh`` — from two exponentials; ``exprel`` — a quartic series near zero,
  ``(e^x - 1)/x`` elsewhere; ``x**(1/3)`` — ``2^(log2|x| / 3)`` with the sign of ``x``.

Every intermediate stays within a signed 64-bit integer for words up to 32 bits,
so the C/Rust kernel can mirror this module with ``int64_t`` and the RTL with
64-bit signed registers; :func:`check_format` refuses wider words. Divisions
truncate towards zero and right shifts floor, as in those targets. Results are
signed codes of the format, saturated to the word. Inputs outside the Python
model's domain have a stated value instead of an error: ``log`` of ``x <= 0`` is
the most negative code, ``sqrt`` of ``x < 0`` is 0, and ``x**(1/3)`` of a
negative ``x`` is ``-(|x|**(1/3))``.

The approximation error of every unit against the Python model is measured by
:mod:`sc_neurocore.compiler.lut_accuracy` with ``lowering="v2"``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cache

LOWERING = "v2"
P = 30  # internal fraction bits
B = 30  # bits of a reduced argument on [0, 1) (exp, log) or of one turn (sin, cos)
# Quadratic segments per table: chosen as the smallest counts that keep every
# unit within its stated error (tests/test_function_units.py measures them).
SEGMENT_BITS = {"exp2": 9, "log2": 9, "qsin": 8, "sigmoid": 11, "atan": 9, "erf": 10}
SIGMOID_RANGE_BITS = 5  # the sigmoid table covers [0, 32)
ERF_RANGE_BITS = 2  # the erf table covers [0, 4); erf(4) = 1 - 1.5e-8
MAX_WIDTH = 32
_ONE = 1 << P


def check_format(data_width: int, fraction: int) -> None:
    """Refuse a format the 64-bit units cannot mirror exactly.

    Parameters
    ----------
    data_width, fraction : int
        The fixed-point format.

    Raises
    ------
    ValueError
        If ``data_width`` is outside ``[2, 32]``, ``fraction`` outside
        ``[0, min(P, data_width - 1)]``, or ``data_width + fraction > 61`` (the
        square root's ``x·2**(fraction+2)`` would leave 63 bits).

    """
    if not 2 <= data_width <= MAX_WIDTH:
        raise ValueError(
            f"function lowering v2 supports words of 2 to {MAX_WIDTH} bits, got {data_width}"
        )
    if not 0 <= fraction <= min(P, data_width - 1):
        raise ValueError(f"function lowering v2 needs 0 <= fraction < data_width, got {fraction}")
    if data_width + fraction > 61:
        raise ValueError(
            f"function lowering v2 needs data_width + fraction <= 61, got {data_width + fraction}"
        )


def shift_round(value: int, shift: int) -> int:
    """Return ``value · 2**shift``, rounding half up when ``shift`` is negative.

    Parameters
    ----------
    value : int
        Signed integer.
    shift : int
        Left shift if non-negative, otherwise a rounded right shift.

    Returns
    -------
    int
        The shifted value; ``floor(value / 2**-shift + 1/2)`` for a right shift.

    """
    if shift >= 0:
        return value << shift
    return (value + (1 << (-shift - 1))) >> -shift


def saturate(value: int, data_width: int) -> int:
    """Clamp ``value`` to the signed range of ``data_width`` bits.

    Parameters
    ----------
    value : int
        Signed integer.
    data_width : int
        Word width.

    Returns
    -------
    int
        ``value`` limited to ``[-2**(data_width-1), 2**(data_width-1) - 1]``.

    """
    high = (1 << (data_width - 1)) - 1
    return max(-high - 1, min(high, value))


def scale_to_code(value: int, shift: int, data_width: int) -> int:
    """Return ``saturate(shift_round(value, shift))`` without forming an over-wide value.

    Parameters
    ----------
    value : int
        Signed integer below ``2**63`` in magnitude.
    shift : int
        Left shift if non-negative, otherwise a rounded right shift.
    data_width : int
        Word width of the result.

    Returns
    -------
    int
        The saturated result code; a left shift that would leave the word saturates
        before it is formed.

    """
    if shift > 0 and abs(value).bit_length() + shift > data_width:
        return saturate(-(1 << data_width) if value < 0 else 1 << data_width, data_width)
    return saturate(shift_round(value, shift), data_width)


def trunc_div(numerator: int, denominator: int) -> int:
    """Return the quotient truncated towards zero, as C and Verilog divide.

    Parameters
    ----------
    numerator, denominator : int
        Signed integers; ``denominator`` is not zero.

    Returns
    -------
    int
        ``sign · (|numerator| // |denominator|)``.

    """
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


@cache
def table(name: str) -> tuple[tuple[int, int, int], ...]:
    """Return the quadratic coefficients ``(a, b, c)`` of each segment, scaled by ``2**P``.

    On segment ``i`` with local position ``t`` in ``[0, 1]`` the unit evaluates
    ``a + b·t + c·t²``, the parabola through the function at the segment's start,
    middle and end: ``a = f0``, ``b = 4·fm - 3·f0 - f1``, ``c = 2·f0 + 2·f1 - 4·fm``.

    Parameters
    ----------
    name : str
        ``"exp2"`` (``2^u`` on ``[0, 1]``), ``"log2"`` (``log2(1 + u)`` on ``[0, 1]``),
        ``"qsin"`` (``sin(π/2·u)`` on ``[0, 1]``), ``"sigmoid"`` (``σ(x)`` on ``[0, 32]``),
        ``"atan"`` (``atan(u)`` on ``[0, 1]``) or ``"erf"`` (``erf(x)`` on ``[0, 4]``).

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        ``2**SEGMENT_BITS[name]`` coefficient triples.

    Raises
    ------
    KeyError
        For an unknown table name.

    """
    functions: dict[str, tuple[Callable[[float], float], float]] = {
        "exp2": (lambda u: 2.0**u, 1.0),
        "log2": (lambda u: math.log2(1.0 + u), 1.0),
        "qsin": (lambda u: math.sin(math.pi / 2 * u), 1.0),
        "sigmoid": (lambda x: 1.0 / (1.0 + math.exp(-x)), float(1 << SIGMOID_RANGE_BITS)),
        "atan": (math.atan, 1.0),
        "erf": (math.erf, float(1 << ERF_RANGE_BITS)),
    }
    function, span = functions[name]
    count = 1 << SEGMENT_BITS[name]
    width = span / count
    coefficients = []
    for i in range(count):
        f0 = function(i * width) * _ONE
        fm = function((i + 0.5) * width) * _ONE
        f1 = function((i + 1) * width) * _ONE
        coefficients.append(
            (round(f0), round(4 * fm - 3 * f0 - f1), round(2 * f0 + 2 * f1 - 4 * fm))
        )
    return tuple(coefficients)


def interpolate(name: str, position: int, position_bits: int) -> int:
    """Evaluate table ``name`` at ``position / 2**position_bits`` of its span.

    Parameters
    ----------
    name : str
        A table of :func:`table`.
    position : int
        ``0 <= position <= 2**position_bits``; the end point is the last segment at ``t = 1``.
    position_bits : int
        Bits of ``position``; at least the table's segment bits.

    Returns
    -------
    int
        ``a + floor((b + floor(c·r / 2**s))·r / 2**s)`` at fraction ``P``, with
        ``i = position >> s``, ``r`` the low ``s`` bits and ``s = position_bits - SEGMENT_BITS[name]``.

    """
    shift = position_bits - SEGMENT_BITS[name]
    index = position >> shift
    rest = position & ((1 << shift) - 1)
    if index == 1 << SEGMENT_BITS[name]:  # the end point of the span
        index, rest = index - 1, 1 << shift
    a, b, c = table(name)[index]
    return a + (((b + ((c * rest) >> shift)) * rest) >> shift)


_LOG2E61 = round(math.log2(math.e) * (1 << (P + 31)))  # log2(e), fraction P + 31
_LOG2E_HIGH = _LOG2E61 >> 31
_LOG2E_LOW = _LOG2E61 & ((1 << 31) - 1)
_LN2 = round(math.log(2.0) * _ONE)
_TURN61 = round((1 << (P + 31)) / (2.0 * math.pi))  # one turn per 2π, fraction P + 31
_TURN_HIGH = _TURN61 >> 31
_TURN_LOW = _TURN61 & ((1 << 31) - 1)
_SIXTH = round(_ONE / 6.0)
_TWENTY_FOURTH = round(_ONE / 24.0)
_HUNDRED_TWENTIETH = round(_ONE / 120.0)
_THIRD = round(_ONE / 3.0)
_LOG10_2 = round(math.log10(2.0) * _ONE)
_HALF_PI = round(math.pi / 2 * _ONE)


def exp_parts(code: int, fraction: int) -> tuple[int, int]:
    """Return ``(k, m)`` with ``e^x = m·2**(k - P)`` and ``2**P <= m < 2**(P+1)``.

    Parameters
    ----------
    code : int
        Signed input code (``|code| <= 2**31``).
    fraction : int
        Fraction bits of the input.

    Returns
    -------
    tuple[int, int]
        The integer part ``k`` of ``x·log2(e)`` and the mantissa ``m = 2^f·2**P``.

    """
    product = code * _LOG2E_HIGH + ((code * _LOG2E_LOW) >> 31)  # x·log2(e), fraction: fraction + P
    whole = product >> (fraction + P)
    rest = product - (whole << (fraction + P))
    return whole, interpolate("exp2", rest >> (fraction + P - B), B)


def exp_scaled(code: int, fraction: int, out_fraction: int) -> int:
    """Return ``round(e^x · 2**out_fraction)`` for input code ``x``, capped at ``2**62``.

    Parameters
    ----------
    code : int
        Signed input code (``|code| <= 2**31``).
    fraction : int
        Fraction bits of the input.
    out_fraction : int
        Fraction bits of the result (``<= fraction + 2``).

    Returns
    -------
    int
        Non-negative integer, at most ``2**62`` (no word saturation).

    """
    whole, mantissa = exp_parts(code, fraction)
    shift = whole + out_fraction - P
    if shift > 62 - (P + 1):
        return 1 << 62
    if shift < -(P + 2):
        return 0
    return min(shift_round(mantissa, shift), 1 << 62)


def exp_capped(code: int, data_width: int, fraction: int) -> int:
    """Return ``round(e^x · 2**fraction)`` for input code ``x``, capped at ``2**62``.

    Parameters
    ----------
    code : int
        Signed input code (``|code| <= 2**31``).
    data_width, fraction : int
        The format (``data_width`` is not used: the result is not saturated).

    Returns
    -------
    int
        Non-negative result code, at most ``2**62`` (no word saturation yet).

    """
    return exp_scaled(code, fraction, fraction)


def exp(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``e^x``, saturated to the word.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(exp_capped(code, data_width, fraction), data_width)


def log2_internal(code: int, fraction: int) -> int:
    """Return ``log2(x)`` at fraction ``P`` for a positive input code ``x``.

    Parameters
    ----------
    code : int
        Positive input code.
    fraction : int
        Fraction bits of the input.

    Returns
    -------
    int
        ``(k - fraction)·2**P + log2(m)·2**P`` with ``code = m·2**k``, ``1 <= m < 2``.

    """
    lead = code.bit_length() - 1
    residue = code - (1 << lead)
    position = residue >> (lead - B) if lead >= B else residue << (B - lead)
    return ((lead - fraction) << P) + interpolate("log2", position, B)


def log(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``ln x``; the most negative code for ``x <= 0``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    if code <= 0:
        return -(1 << (data_width - 1))
    log2x = log2_internal(code, fraction)
    whole = log2x >> P
    rest = log2x - (whole << P)
    natural = whole * _LN2 + ((rest * _LN2) >> P)  # fraction P
    return saturate(shift_round(natural, fraction - P), data_width)


def isqrt(value: int) -> int:
    """Return ``floor(sqrt(value))`` by the restoring digit-by-digit method.

    Parameters
    ----------
    value : int
        Non-negative integer below ``2**63``.

    Returns
    -------
    int
        The integer square root.

    """
    root = 0
    bit = 1 << 62
    while bit > value:
        bit >>= 2
    while bit:
        if value >= root + bit:
            value -= root + bit
            root = (root >> 1) + bit
        else:
            root >>= 1
        bit >>= 2
    return root


def sqrt(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``sqrt(x)`` rounded to nearest; 0 for ``x < 0``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    if code <= 0:
        return 0
    return saturate((isqrt((code << fraction) << 2) + 1) >> 1, data_width)


def _turn_sine(turn: int) -> int:
    """Return ``sin(2π·turn/2**B)`` at fraction ``P`` for ``0 <= turn < 2**B``."""
    quarter_bits = B - 2
    quadrant = turn >> quarter_bits
    within = turn & ((1 << quarter_bits) - 1)
    if quadrant & 1:
        within = (1 << quarter_bits) - within
    value = interpolate("qsin", within, quarter_bits)
    return -value if quadrant >= 2 else value


def _turns(code: int, fraction: int) -> int:
    """Return the input as a fraction of one turn, ``B`` bits, reduced modulo one turn."""
    product = code * _TURN_HIGH + ((code * _TURN_LOW) >> 31)  # turns, fraction: fraction + P
    return (product & ((1 << (fraction + P)) - 1)) >> (fraction + P - B)


def sin(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``sin x``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(shift_round(_turn_sine(_turns(code, fraction)), fraction - P), data_width)


def cos(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``cos x``, as the sine a quarter turn ahead.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    turn = (_turns(code, fraction) + (1 << (B - 2))) & ((1 << B) - 1)
    return saturate(shift_round(_turn_sine(turn), fraction - P), data_width)


def sigmoid_internal(code: int, fraction: int) -> int:
    """Return ``σ(x)`` at fraction ``P`` for any signed input code.

    Parameters
    ----------
    code : int
        Signed input code (any magnitude below ``2**33``).
    fraction : int
        Fraction bits of the input.

    Returns
    -------
    int
        ``σ(x)·2**P``; ``2**P`` at and beyond ``|x| = 32``.

    """
    magnitude = -code if code < 0 else code
    position_bits = fraction + SIGMOID_RANGE_BITS
    if magnitude >= 1 << position_bits:
        value = _ONE
    elif position_bits >= SEGMENT_BITS["sigmoid"]:
        value = interpolate("sigmoid", magnitude, position_bits)
    else:
        value = interpolate(
            "sigmoid",
            magnitude << (SEGMENT_BITS["sigmoid"] - position_bits),
            SEGMENT_BITS["sigmoid"],
        )
    return _ONE - value if code < 0 else value


def sigmoid(code: int, data_width: int, fraction: int) -> int:
    """Return the code of the logistic function ``1/(1 + e^-x)``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(shift_round(sigmoid_internal(code, fraction), fraction - P), data_width)


def tanh(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``tanh x = 2σ(2x) - 1``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    value = 2 * sigmoid_internal(2 * code, fraction) - _ONE
    return saturate(shift_round(value, fraction - P), data_width)


def cosh(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``cosh x = (e^x + e^-x)/2``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    # Both terms carry two extra bits, so the sum is rounded once.
    total = exp_scaled(code, fraction, fraction + 2) + exp_scaled(-code, fraction, fraction + 2)
    return saturate(shift_round(total, -3), data_width)


def sinh(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``sinh x = (e^x - e^-x)/2``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    # Both terms carry two extra bits, so the difference is rounded once.
    difference = exp_scaled(code, fraction, fraction + 2) - exp_scaled(
        -code, fraction, fraction + 2
    )
    return saturate(shift_round(difference, -3), data_width)


def exprel(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``exprel x = (e^x - 1)/x``, with ``exprel 0 = 1``.

    For ``|x| < 1/8`` the series ``1 + x/2 + x²/6 + x³/24 + x⁴/120`` (truncation
    below ``|x|⁵/720 < 4.3·10⁻⁸``); otherwise ``(e^x - 1)/x`` with a division truncated
    towards zero.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    one = 1 << fraction
    if -(one >> 3) < code < one >> 3 or one >> 3 == 0 and code == 0:
        z = code << (P - fraction)  # fraction P
        z2 = (z * z) >> P
        z3 = (z2 * z) >> P
        z4 = (z3 * z) >> P
        series = (
            _ONE
            + (z >> 1)
            + ((z2 * _SIXTH) >> P)
            + ((z3 * _TWENTY_FOURTH) >> P)
            + ((z4 * _HUNDRED_TWENTIETH) >> P)
        )
        return saturate(shift_round(series, fraction - P), data_width)
    whole, mantissa = exp_parts(code, fraction)
    if whole > 30:
        # e^x > 2**30: the -1 changes the quotient by less than 2**-30 of it.
        quotient = trunc_div(mantissa << 31, code)  # fraction P + 31 - fraction - whole
        return scale_to_code(quotient, 2 * fraction - P - 31 + whole, data_width)
    numerator = shift_round(mantissa, whole) - _ONE  # e^x - 1, fraction P, |numerator| < 2**61
    spare = max(0, 61 - abs(numerator).bit_length())
    quotient = trunc_div(numerator << spare, code)  # fraction P + spare - fraction
    return scale_to_code(quotient, 2 * fraction - P - spare, data_width)


def cbrt(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``x**(1/3)``, odd in ``x``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    if code == 0:
        return 0
    magnitude = -code if code < 0 else code
    log2x = log2_internal(magnitude, fraction)
    high = log2x >> P
    third = high * _THIRD + (
        ((log2x - (high << P)) * _THIRD) >> P
    )  # fraction P, split to stay in 63 bits
    whole = third >> P
    rest = third - (whole << P)
    mantissa = interpolate("exp2", rest >> (P - B), B)
    value = scale_to_code(mantissa, whole + fraction - P, data_width)
    return -value if code < 0 else value


def _code_to_internal(code: int, fraction: int) -> int:
    """Return the input at fraction ``P`` (``fraction <= P``)."""
    return code << (P - fraction)


def exp2(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``2**x``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    whole = code >> fraction
    rest = code - (whole << fraction)
    position = rest << (B - fraction)
    mantissa = interpolate("exp2", position, B)
    return scale_to_code(mantissa, whole + fraction - P, data_width)


def log2(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``log2 x``; the most negative code for ``x <= 0``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    if code <= 0:
        return -(1 << (data_width - 1))
    return saturate(shift_round(log2_internal(code, fraction), fraction - P), data_width)


def log10(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``log10 x = log2 x · log10 2``; the most negative code for ``x <= 0``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    if code <= 0:
        return -(1 << (data_width - 1))
    log2x = log2_internal(code, fraction)
    whole = log2x >> P
    rest = log2x - (whole << P)
    value = whole * _LOG10_2 + ((rest * _LOG10_2) >> P)
    return saturate(shift_round(value, fraction - P), data_width)


def expm1(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``e^x - 1``: the exponential's rounded code minus an exact one.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(exp_capped(code, data_width, fraction) - (1 << fraction), data_width)


def log1p(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``log(1 + x)``, from the exact integer ``1 + x``; most negative for ``x <= -1``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code, saturated to the word.

    """
    if code <= -(1 << fraction):
        return -(1 << (data_width - 1))
    return log(code + (1 << fraction), data_width, fraction)


def tan(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``tan x = sin x / cos x``, rounded half away from zero; saturated at the poles.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    turn = _turns(code, fraction)
    sine = _turn_sine(turn)
    cosine = _turn_sine((turn + (1 << (B - 2))) & ((1 << B) - 1))
    if cosine == 0:
        return saturate(-(1 << data_width) if sine < 0 else 1 << data_width, data_width)
    doubled = trunc_div(sine << (fraction + 1), cosine)
    return saturate(trunc_div(doubled + (1 if doubled >= 0 else -1), 2), data_width)


def _atan_internal(numerator: int, denominator: int) -> int:
    """Return ``atan(numerator / denominator)`` at fraction ``P`` for ``0 <= numerator``, ``0 < denominator``."""
    if numerator <= denominator:
        return interpolate("atan", trunc_div(numerator << P, denominator), P)
    return _HALF_PI - interpolate("atan", trunc_div(denominator << P, numerator), P)


def arctan(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``arctan x``, with ``arctan x = π/2 - arctan(1/x)`` above 1.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    magnitude = -code if code < 0 else code
    value = _atan_internal(magnitude, 1 << fraction)
    value = -value if code < 0 else value
    return saturate(shift_round(value, fraction - P), data_width)


def atan2(y: int, x: int, data_width: int, fraction: int) -> int:
    """Return the saturated code of the quadrant-aware angle ``atan2(y, x)``.

    Parameters
    ----------
    y, x : int
        Signed codes, respectively vertical and horizontal coordinates.
        Fixed-point zero has the positive-zero convention; the origin returns 0.
    data_width, fraction : int
        The common input and output format, checked by :func:`evaluate`.

    Returns
    -------
    int
        Angle in radians, rounded half up and saturated to the signed word.
        The conservative arithmetic error bound against the clipped real angle
        is ``0.5 * 2**-fraction + 8 * 2**-P``; see the accuracy report.

    Notes
    -----
    Reduces to the existing quadratic arctangent table on [0, 1]. No argument
    ratio is formed in floating point. IEEE signed zeros and infinities are not
    representable input codes. NumPy documents the operand order at
    https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html.
    """
    if y == 0:
        value = 2 * _HALF_PI if x < 0 else 0
    elif x == 0:
        value = _HALF_PI
    else:
        value = _atan_internal(abs(y), abs(x))
        if x < 0:
            value = 2 * _HALF_PI - value
    if y < 0:
        value = -value
    return saturate(shift_round(value, fraction - P), data_width)


def _arcsin_internal(code: int, fraction: int) -> int:
    """Return ``arcsin x`` at fraction ``P``, ``±π/2`` for ``|x| >= 1``."""
    magnitude = min(-code if code < 0 else code, 1 << fraction)
    x = _code_to_internal(magnitude, fraction)
    root = isqrt((1 << (2 * P)) - x * x)  # sqrt(1 - x²), fraction P
    value = _HALF_PI if root == 0 else _atan_internal(x, root)
    return -value if code < 0 else value


def arcsin(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``arcsin x = arctan(x / sqrt(1 - x²))``; ``±π/2`` for ``|x| >= 1``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(shift_round(_arcsin_internal(code, fraction), fraction - P), data_width)


def arccos(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``arccos x = π/2 - arcsin x``; ``0`` or ``π`` for ``|x| >= 1``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(
        shift_round(_HALF_PI - _arcsin_internal(code, fraction), fraction - P), data_width
    )


def erf(code: int, data_width: int, fraction: int) -> int:
    """Return the code of the error function, odd in ``x``, 1 at and beyond ``|x| = 4``.

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    magnitude = -code if code < 0 else code
    position_bits = fraction + ERF_RANGE_BITS
    bits = SEGMENT_BITS["erf"]
    if magnitude >= 1 << position_bits:
        value = _ONE
    elif position_bits >= bits:
        value = interpolate("erf", magnitude, position_bits)
    else:
        value = interpolate("erf", magnitude << (bits - position_bits), bits)
    value = -value if code < 0 else value
    return saturate(shift_round(value, fraction - P), data_width)


def floor(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``floor x`` (exact).

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return (code >> fraction) << fraction


def ceil(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``ceil x`` (exact; saturated where the next integer leaves the word).

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return saturate(-((-code) >> fraction << fraction), data_width)


def trunc(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``trunc x``, rounding towards zero (exact).

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    return floor(code, data_width, fraction) if code >= 0 else ceil(code, data_width, fraction)


def sign(code: int, data_width: int, fraction: int) -> int:
    """Return the code of ``sign x`` (-1, 0 or 1; 1 saturates where the word cannot hold it).

    Parameters
    ----------
    code : int
        Signed input code.
    data_width, fraction : int
        The format.

    Returns
    -------
    int
        The result code.

    """
    unit = (code > 0) - (code < 0)
    return saturate(unit << fraction, data_width)


UNITS: dict[str, Callable[[int, int, int], int]] = {
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "cosh": cosh,
    "sinh": sinh,
    "exprel": exprel,
    "cbrt": cbrt,
    "exp2": exp2,
    "log2": log2,
    "log10": log10,
    "expm1": expm1,
    "log1p": log1p,
    "tan": tan,
    "arctan": arctan,
    "arcsin": arcsin,
    "arccos": arccos,
    "erf": erf,
    "floor": floor,
    "ceil": ceil,
    "trunc": trunc,
    "sign": sign,
}

BINARY_UNITS: dict[str, Callable[[int, int, int, int], int]] = {"atan2": atan2}
UNARY_FUNCTIONS = tuple(UNITS)
FUNCTIONS = (*UNARY_FUNCTIONS, *BINARY_UNITS)
FUNCTION_ARITY = {name: 1 for name in UNITS} | {name: 2 for name in BINARY_UNITS}


def evaluate(
    function: str,
    code: int,
    data_width: int,
    fraction: int,
    *,
    additional_inputs: Sequence[int] = (),
) -> int:
    """Evaluate one ``v2`` unit on its signed input codes.

    Parameters
    ----------
    function : str
        A member of :data:`FUNCTIONS`.
    code : int
        Signed input code of the format.
    data_width, fraction : int
        The format.
    additional_inputs : sequence of int, optional
        Further operands in order; ``atan2`` takes its horizontal coordinate here.

    Returns
    -------
    int
        The result code.

    Raises
    ------
    ValueError
        For an unknown function, wrong operand count, a format
        :func:`check_format` refuses, or binary input codes outside the word.

    """
    if function not in FUNCTION_ARITY:
        raise ValueError(f"no v2 unit for {function!r}")
    check_format(data_width, fraction)
    if 1 + len(additional_inputs) != FUNCTION_ARITY[function]:
        raise ValueError(f"{function} needs exactly {FUNCTION_ARITY[function]} input code(s)")
    if function in BINARY_UNITS:
        low, high = -(1 << (data_width - 1)), (1 << (data_width - 1)) - 1
        if any(not low <= value <= high for value in (code, *additional_inputs)):
            raise ValueError(f"{function} input code outside signed {data_width}-bit domain")
        return BINARY_UNITS[function](code, additional_inputs[0], data_width, fraction)
    return UNITS[function](code, data_width, fraction)


__all__ = [
    "B",
    "FUNCTIONS",
    "FUNCTION_ARITY",
    "UNARY_FUNCTIONS",
    "LOWERING",
    "MAX_WIDTH",
    "P",
    "check_format",
    "evaluate",
    "isqrt",
    "exp_parts",
    "exp_scaled",
    "interpolate",
    "saturate",
    "scale_to_code",
    "shift_round",
    "table",
    "trunc_div",
]
