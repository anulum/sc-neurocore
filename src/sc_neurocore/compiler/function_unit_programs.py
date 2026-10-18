# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — function lowering v2: each unit as a 64-bit integer program

"""Build the ``v2`` function units as :class:`~sc_neurocore.compiler.function_unit_ir.Program` values.

Each builder follows the reference arithmetic of
:mod:`sc_neurocore.compiler.function_units` step for step; the tests evaluate
both on every input code of 16-bit formats and on sampled codes of 32-bit
formats and require equal results. Because a select computes both of its
operands, every branch receives an input already limited to that branch's
domain (for example the series of ``exprel`` sees 0 outside ``|x| < 1/8``), so no
step of an unselected branch leaves 64 bits.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

from . import function_units as fu
from .function_unit_ir import Program

Builder = Callable[[Program, str, int, int], str]


def _shift_round(p: Program, value: str, shift: int) -> str:
    if shift >= 0:
        return p.shl(value, shift)
    return p.shr(p.add(value, p.const(1 << (-shift - 1))), -shift)


def _shift_round_var(p: Program, value: str, shift: str) -> str:
    """``shift_round`` with a computed shift; both directions are formed, one is selected.

    A right shift beyond 62 bits is taken as 62: every value shifted here is below
    ``2**61`` in magnitude, so both round to the same result and ``2**63`` is never formed.
    """
    zero = p.const(0)
    negative_shift = _min(p, p.neg(shift), p.const(62))
    half = p.shl_var(p.const(1), p.sub(negative_shift, p.const(1)))
    right = p.shr_var(p.add(value, half), negative_shift)
    return p.select(p.le(zero, shift), p.shl_var(value, shift), right)


def _saturate(p: Program, value: str, data_width: int) -> str:
    high = p.const((1 << (data_width - 1)) - 1)
    low = p.const(-(1 << (data_width - 1)))
    return p.select(p.lt(value, low), low, p.select(p.lt(high, value), high, value))


def _abs(p: Program, value: str) -> str:
    return p.select(p.lt(value, p.const(0)), p.neg(value), value)


def _max(p: Program, a: str, b: str) -> str:
    return p.select(p.lt(a, b), b, a)


def _min(p: Program, a: str, b: str) -> str:
    return p.select(p.lt(b, a), b, a)


def _bit_length(p: Program, value: str) -> str:
    """Bits of a non-negative value below ``2**63``, by an unrolled binary search."""
    rest, count = value, p.const(0)
    for width in (32, 16, 8, 4, 2, 1):
        big = p.le(p.const(1 << width), rest)
        rest = p.select(big, p.shr(rest, width), rest)
        count = p.select(big, p.add(count, p.const(width)), count)
    return p.add(count, p.lt(p.const(0), rest))


def _isqrt(p: Program, value: str) -> str:
    """Restoring square root over all 32 digit positions (identical to skipping leading zeros)."""
    root, rest = p.const(0), value
    for position in range(62, -1, -2):
        bit = p.const(1 << position)
        trial = p.add(root, bit)
        take = p.le(trial, rest)
        rest = p.select(take, p.sub(rest, trial), rest)
        root = p.select(take, p.add(p.shr(root, 1), bit), p.shr(root, 1))
    return root


def _scale_to_code(p: Program, value: str, shift: str, data_width: int) -> str:
    """``scale_to_code`` with the left shift limited before it is formed."""
    magnitude_bits = _bit_length(p, _abs(p, value))
    overflow = p.mul(
        p.lt(p.const(0), shift), p.lt(p.const(data_width), p.add(magnitude_bits, shift))
    )
    extreme = p.select(
        p.lt(value, p.const(0)),
        p.const(-(1 << (data_width - 1))),
        p.const((1 << (data_width - 1)) - 1),
    )
    safe_shift = p.select(overflow, p.const(0), shift)
    return p.select(
        overflow, extreme, _saturate(p, _shift_round_var(p, value, safe_shift), data_width)
    )


def _interpolate(p: Program, name: str, position: str, position_bits: int) -> str:
    bits = fu.SEGMENT_BITS[name]
    shift = position_bits - bits
    index = p.shr(position, shift)
    rest = p.mask(position, shift)
    at_end = p.eq(index, p.const(1 << bits))
    index = p.select(at_end, p.const((1 << bits) - 1), index)
    rest = p.select(at_end, p.const(1 << shift), rest)
    a, b, c = (p.lookup(name, index, k) for k in range(3))
    inner = p.add(b, p.shr(p.mul(c, rest), shift))
    return p.add(a, p.shr(p.mul(inner, rest), shift))


def _exp_parts(p: Program, code: str, fraction: int) -> tuple[str, str]:
    product = p.add(
        p.mul(code, p.const(fu._LOG2E_HIGH)), p.shr(p.mul(code, p.const(fu._LOG2E_LOW)), 31)
    )
    whole = p.shr(product, fraction + fu.P)
    rest = p.sub(product, p.shl(whole, fraction + fu.P))
    return whole, _interpolate(p, "exp2", p.shr(rest, fraction + fu.P - fu.B), fu.B)


def _exp_scaled(p: Program, code: str, fraction: int, out_fraction: int) -> str:
    whole, mantissa = _exp_parts(p, code, fraction)
    shift = p.add(whole, p.const(out_fraction - fu.P))
    top, bottom = 62 - (fu.P + 1), -(fu.P + 2)
    bounded = _min(p, _max(p, shift, p.const(bottom)), p.const(top))
    scaled = _min(p, _shift_round_var(p, mantissa, bounded), p.const(1 << 62))
    return p.select(
        p.lt(p.const(top), shift),
        p.const(1 << 62),
        p.select(p.lt(shift, p.const(bottom)), p.const(0), scaled),
    )


def _log2_internal(p: Program, code: str, fraction: int) -> str:
    lead = p.sub(_bit_length(p, code), p.const(1))
    residue = p.sub(code, p.shl_var(p.const(1), lead))
    above = p.le(p.const(fu.B), lead)
    position = p.select(
        above,
        p.shr_var(residue, p.sub(lead, p.const(fu.B))),
        p.shl_var(residue, p.sub(p.const(fu.B), lead)),
    )
    return p.add(
        p.shl(p.sub(lead, p.const(fraction)), fu.P), _interpolate(p, "log2", position, fu.B)
    )


def _unit_exp(p: Program, code: str, data_width: int, fraction: int) -> str:
    return _saturate(p, _exp_scaled(p, code, fraction, fraction), data_width)


def _unit_log(p: Program, code: str, data_width: int, fraction: int) -> str:
    positive = p.lt(p.const(0), code)
    log2x = _log2_internal(p, p.select(positive, code, p.const(1)), fraction)
    whole = p.shr(log2x, fu.P)
    rest = p.sub(log2x, p.shl(whole, fu.P))
    natural = p.add(p.mul(whole, p.const(fu._LN2)), p.shr(p.mul(rest, p.const(fu._LN2)), fu.P))
    value = _saturate(p, _shift_round(p, natural, fraction - fu.P), data_width)
    return p.select(positive, value, p.const(-(1 << (data_width - 1))))


def _unit_sqrt(p: Program, code: str, data_width: int, fraction: int) -> str:
    positive = p.lt(p.const(0), code)
    radicand = p.shl(p.select(positive, code, p.const(0)), fraction + 2)
    root = p.shr(p.add(_isqrt(p, radicand), p.const(1)), 1)
    return p.select(positive, _saturate(p, root, data_width), p.const(0))


def _turns(p: Program, code: str, fraction: int) -> str:
    product = p.add(
        p.mul(code, p.const(fu._TURN_HIGH)), p.shr(p.mul(code, p.const(fu._TURN_LOW)), 31)
    )
    return p.shr(p.mask(product, fraction + fu.P), fraction + fu.P - fu.B)


def _turn_sine(p: Program, turn: str) -> str:
    quarter_bits = fu.B - 2
    quadrant = p.shr(turn, quarter_bits)
    within = p.mask(turn, quarter_bits)
    odd = p.mask(quadrant, 1)
    within = p.select(odd, p.sub(p.const(1 << quarter_bits), within), within)
    value = _interpolate(p, "qsin", within, quarter_bits)
    return p.select(p.le(p.const(2), quadrant), p.neg(value), value)


def _unit_sin(p: Program, code: str, data_width: int, fraction: int) -> str:
    sine = _turn_sine(p, _turns(p, code, fraction))
    return _saturate(p, _shift_round(p, sine, fraction - fu.P), data_width)


def _unit_cos(p: Program, code: str, data_width: int, fraction: int) -> str:
    turn = p.mask(p.add(_turns(p, code, fraction), p.const(1 << (fu.B - 2))), fu.B)
    return _saturate(p, _shift_round(p, _turn_sine(p, turn), fraction - fu.P), data_width)


def _sigmoid_internal(p: Program, code: str, fraction: int) -> str:
    one = p.const(1 << fu.P)
    negative = p.lt(code, p.const(0))
    magnitude = _abs(p, code)
    position_bits = fraction + fu.SIGMOID_RANGE_BITS
    bits = fu.SEGMENT_BITS["sigmoid"]
    inside = p.lt(magnitude, p.const(1 << position_bits))
    bounded = p.select(inside, magnitude, p.const(0))
    if position_bits >= bits:
        table_value = _interpolate(p, "sigmoid", bounded, position_bits)
    else:
        table_value = _interpolate(p, "sigmoid", p.shl(bounded, bits - position_bits), bits)
    value = p.select(inside, table_value, one)
    return p.select(negative, p.sub(one, value), value)


def _unit_sigmoid(p: Program, code: str, data_width: int, fraction: int) -> str:
    return _saturate(
        p, _shift_round(p, _sigmoid_internal(p, code, fraction), fraction - fu.P), data_width
    )


def _unit_tanh(p: Program, code: str, data_width: int, fraction: int) -> str:
    doubled = p.sub(p.shl(_sigmoid_internal(p, p.shl(code, 1), fraction), 1), p.const(1 << fu.P))
    return _saturate(p, _shift_round(p, doubled, fraction - fu.P), data_width)


def _unit_cosh(p: Program, code: str, data_width: int, fraction: int) -> str:
    total = p.add(
        _exp_scaled(p, code, fraction, fraction + 2),
        _exp_scaled(p, p.neg(code), fraction, fraction + 2),
    )
    return _saturate(p, _shift_round(p, total, -3), data_width)


def _unit_sinh(p: Program, code: str, data_width: int, fraction: int) -> str:
    difference = p.sub(
        _exp_scaled(p, code, fraction, fraction + 2),
        _exp_scaled(p, p.neg(code), fraction, fraction + 2),
    )
    return _saturate(p, _shift_round(p, difference, -3), data_width)


def _unit_exprel(p: Program, code: str, data_width: int, fraction: int) -> str:
    one_code = 1 << fraction
    near = p.mul(p.lt(p.const(-(one_code >> 3)), code), p.lt(code, p.const(one_code >> 3)))
    if one_code >> 3 == 0:
        near = p.eq(code, p.const(0))
    series_input = p.select(near, code, p.const(0))
    z = p.shl(series_input, fu.P - fraction)
    z2 = p.shr(p.mul(z, z), fu.P)
    z3 = p.shr(p.mul(z2, z), fu.P)
    z4 = p.shr(p.mul(z3, z), fu.P)
    series = p.add(p.const(1 << fu.P), p.shr(z, 1))
    series = p.add(series, p.shr(p.mul(z2, p.const(fu._SIXTH)), fu.P))
    series = p.add(series, p.shr(p.mul(z3, p.const(fu._TWENTY_FOURTH)), fu.P))
    series = p.add(series, p.shr(p.mul(z4, p.const(fu._HUNDRED_TWENTIETH)), fu.P))
    near_value = _saturate(p, _shift_round(p, series, fraction - fu.P), data_width)

    whole, mantissa = _exp_parts(p, code, fraction)
    large = p.lt(p.const(30), whole)
    large_quotient = p.div(p.shl(mantissa, 31), code)
    large_value = _scale_to_code(
        p, large_quotient, p.add(whole, p.const(2 * fraction - fu.P - 31)), data_width
    )
    numerator = p.sub(
        _shift_round_var(p, mantissa, _min(p, whole, p.const(30))), p.const(1 << fu.P)
    )
    spare = _max(p, p.const(0), p.sub(p.const(61), _bit_length(p, _abs(p, numerator))))
    quotient = p.div(p.shl_var(numerator, spare), code)
    middle_value = _scale_to_code(
        p, quotient, p.sub(p.const(2 * fraction - fu.P), spare), data_width
    )
    return p.select(near, near_value, p.select(large, large_value, middle_value))


def _unit_cbrt(p: Program, code: str, data_width: int, fraction: int) -> str:
    negative = p.lt(code, p.const(0))
    magnitude = _abs(p, code)
    nonzero = p.lt(p.const(0), magnitude)
    log2x = _log2_internal(p, p.select(nonzero, magnitude, p.const(1)), fraction)
    high = p.shr(log2x, fu.P)
    third = p.add(
        p.mul(high, p.const(fu._THIRD)),
        p.shr(p.mul(p.sub(log2x, p.shl(high, fu.P)), p.const(fu._THIRD)), fu.P),
    )
    whole = p.shr(third, fu.P)
    rest = p.sub(third, p.shl(whole, fu.P))
    mantissa = _interpolate(p, "exp2", p.shr(rest, fu.P - fu.B), fu.B)
    value = _scale_to_code(p, mantissa, p.add(whole, p.const(fraction - fu.P)), data_width)
    signed = p.select(negative, p.neg(value), value)
    return p.select(nonzero, signed, p.const(0))


def _unit_exp2(p: Program, code: str, data_width: int, fraction: int) -> str:
    whole = p.shr(code, fraction)
    position = p.shl(p.sub(code, p.shl(whole, fraction)), fu.B - fraction)
    mantissa = _interpolate(p, "exp2", position, fu.B)
    return _scale_to_code(p, mantissa, p.add(whole, p.const(fraction - fu.P)), data_width)


def _unit_log2(p: Program, code: str, data_width: int, fraction: int) -> str:
    positive = p.lt(p.const(0), code)
    log2x = _log2_internal(p, p.select(positive, code, p.const(1)), fraction)
    value = _saturate(p, _shift_round(p, log2x, fraction - fu.P), data_width)
    return p.select(positive, value, p.const(-(1 << (data_width - 1))))


def _unit_log10(p: Program, code: str, data_width: int, fraction: int) -> str:
    positive = p.lt(p.const(0), code)
    log2x = _log2_internal(p, p.select(positive, code, p.const(1)), fraction)
    whole = p.shr(log2x, fu.P)
    rest = p.sub(log2x, p.shl(whole, fu.P))
    value = p.add(
        p.mul(whole, p.const(fu._LOG10_2)), p.shr(p.mul(rest, p.const(fu._LOG10_2)), fu.P)
    )
    rounded = _saturate(p, _shift_round(p, value, fraction - fu.P), data_width)
    return p.select(positive, rounded, p.const(-(1 << (data_width - 1))))


def _unit_expm1(p: Program, code: str, data_width: int, fraction: int) -> str:
    grown = _exp_scaled(p, code, fraction, fraction)
    return _saturate(p, p.sub(grown, p.const(1 << fraction)), data_width)


def _unit_log1p(p: Program, code: str, data_width: int, fraction: int) -> str:
    return _unit_log(p, p.add(code, p.const(1 << fraction)), data_width, fraction)


def _unit_tan(p: Program, code: str, data_width: int, fraction: int) -> str:
    turn = _turns(p, code, fraction)
    sine = _turn_sine(p, turn)
    cosine = _turn_sine(p, p.mask(p.add(turn, p.const(1 << (fu.B - 2))), fu.B))
    doubled = p.div(p.shl(sine, fraction + 1), cosine)
    away = p.select(p.le(p.const(0), doubled), p.const(1), p.const(-1))
    value = _saturate(p, p.div(p.add(doubled, away), p.const(2)), data_width)
    pole = p.select(
        p.lt(sine, p.const(0)),
        p.const(-(1 << (data_width - 1))),
        p.const((1 << (data_width - 1)) - 1),
    )
    return p.select(p.eq(cosine, p.const(0)), pole, value)


def _atan_internal(p: Program, numerator: str, denominator: str) -> str:
    below = p.le(numerator, denominator)
    direct = _interpolate(p, "atan", p.div(p.shl(numerator, fu.P), denominator), fu.P)
    inverse = _interpolate(p, "atan", p.div(p.shl(denominator, fu.P), numerator), fu.P)
    return p.select(below, direct, p.sub(p.const(fu._HALF_PI), inverse))


def _unit_arctan(p: Program, code: str, data_width: int, fraction: int) -> str:
    value = _atan_internal(p, _abs(p, code), p.const(1 << fraction))
    signed = p.select(p.lt(code, p.const(0)), p.neg(value), value)
    return _saturate(p, _shift_round(p, signed, fraction - fu.P), data_width)


def _arcsin_internal(p: Program, code: str, fraction: int) -> str:
    magnitude = _min(p, _abs(p, code), p.const(1 << fraction))
    x = p.shl(magnitude, fu.P - fraction)
    root = _isqrt(p, p.sub(p.const(1 << (2 * fu.P)), p.mul(x, x)))
    value = p.select(p.eq(root, p.const(0)), p.const(fu._HALF_PI), _atan_internal(p, x, root))
    return p.select(p.lt(code, p.const(0)), p.neg(value), value)


def _unit_arcsin(p: Program, code: str, data_width: int, fraction: int) -> str:
    return _saturate(
        p, _shift_round(p, _arcsin_internal(p, code, fraction), fraction - fu.P), data_width
    )


def _unit_arccos(p: Program, code: str, data_width: int, fraction: int) -> str:
    value = p.sub(p.const(fu._HALF_PI), _arcsin_internal(p, code, fraction))
    return _saturate(p, _shift_round(p, value, fraction - fu.P), data_width)


def _unit_erf(p: Program, code: str, data_width: int, fraction: int) -> str:
    magnitude = _abs(p, code)
    position_bits = fraction + fu.ERF_RANGE_BITS
    bits = fu.SEGMENT_BITS["erf"]
    inside = p.lt(magnitude, p.const(1 << position_bits))
    bounded = p.select(inside, magnitude, p.const(0))
    if position_bits >= bits:
        table_value = _interpolate(p, "erf", bounded, position_bits)
    else:
        table_value = _interpolate(p, "erf", p.shl(bounded, bits - position_bits), bits)
    value = p.select(inside, table_value, p.const(1 << fu.P))
    signed = p.select(p.lt(code, p.const(0)), p.neg(value), value)
    return _saturate(p, _shift_round(p, signed, fraction - fu.P), data_width)


def _unit_floor(p: Program, code: str, data_width: int, fraction: int) -> str:
    return p.shl(p.shr(code, fraction), fraction)


def _unit_ceil(p: Program, code: str, data_width: int, fraction: int) -> str:
    return _saturate(p, p.neg(p.shl(p.shr(p.neg(code), fraction), fraction)), data_width)


def _unit_trunc(p: Program, code: str, data_width: int, fraction: int) -> str:
    return p.select(
        p.le(p.const(0), code),
        _unit_floor(p, code, data_width, fraction),
        _unit_ceil(p, code, data_width, fraction),
    )


def _unit_sign(p: Program, code: str, data_width: int, fraction: int) -> str:
    unit = p.sub(p.lt(p.const(0), code), p.lt(code, p.const(0)))
    return _saturate(p, p.shl(unit, fraction), data_width)


_BUILDERS: dict[str, Builder] = {
    "exp": _unit_exp,
    "log": _unit_log,
    "sqrt": _unit_sqrt,
    "sin": _unit_sin,
    "cos": _unit_cos,
    "sigmoid": _unit_sigmoid,
    "tanh": _unit_tanh,
    "cosh": _unit_cosh,
    "sinh": _unit_sinh,
    "exprel": _unit_exprel,
    "cbrt": _unit_cbrt,
    "exp2": _unit_exp2,
    "log2": _unit_log2,
    "log10": _unit_log10,
    "expm1": _unit_expm1,
    "log1p": _unit_log1p,
    "tan": _unit_tan,
    "arctan": _unit_arctan,
    "arcsin": _unit_arcsin,
    "arccos": _unit_arccos,
    "erf": _unit_erf,
    "floor": _unit_floor,
    "ceil": _unit_ceil,
    "trunc": _unit_trunc,
    "sign": _unit_sign,
}


@cache
def program(function: str, data_width: int, fraction: int) -> Program:
    """Return the ``v2`` unit for one function and format as an integer program.

    Parameters
    ----------
    function : str
        A member of :data:`sc_neurocore.compiler.function_units.FUNCTIONS`.
    data_width, fraction : int
        The format.

    Returns
    -------
    Program
        The program; its input is the signed input code, its output the result code.

    Raises
    ------
    ValueError
        For an unknown function or a format the units cannot mirror.

    """
    if function not in _BUILDERS:
        raise ValueError(f"no v2 unit for {function!r}")
    fu.check_format(data_width, fraction)
    p = Program(tables={name: fu.table(name) for name in fu.SEGMENT_BITS})
    p.output = _BUILDERS[function](p, p.input(), data_width, fraction)
    return p


__all__ = ["program"]
