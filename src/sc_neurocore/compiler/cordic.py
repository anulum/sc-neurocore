# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point circular CORDIC arithmetic

"""Define the rounding, range reduction and recurrence of the clocked circular unit."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from functools import cache

from .function_units import check_format, saturate, shift_round

_GUARD_BITS = 10
_RANGE_GUARD_BITS = 44
_ROTATION_SETUP_EDGES = 3
_ROTATION_FUNCTIONS = frozenset({"sin", "cos"})
_FUNCTIONS = _ROTATION_FUNCTIONS | {"arctan", "atan2"}


@dataclass(frozen=True)
class _CordicConstants:
    """Hold integer constants shared by the oracle and emitted RTL."""

    angles: tuple[int, ...]
    inverse_gain: int
    pi: int
    half_pi: int
    wide_pi: int


def _arctan(value: Decimal) -> Decimal:
    """Evaluate the alternating arctangent series for arguments at most one half."""
    square = -(value * value)
    term = value
    total = value
    denominator = 3
    while True:
        term *= square
        updated = total + term / denominator
        if updated == total:
            return total
        total = updated
        denominator += 2


def _fixed(value: Decimal, fraction: int) -> int:
    """Encode a nonnegative constant with the unit's half-up rounding."""
    return int((value * (1 << fraction)).to_integral_value(rounding=ROUND_HALF_UP))


@cache
def _constants(fraction: int) -> _CordicConstants:
    """Construct the geometry-dependent circular constants with ample precision."""
    internal = fraction + _GUARD_BITS
    with localcontext() as ctx:
        ctx.prec = 85
        pi = 16 * _arctan(Decimal(1) / 5) - 4 * _arctan(Decimal(1) / 239)
        gain = Decimal(1)
        angles = [_fixed(pi / 4, internal)]
        for index in range(internal + 1):
            shift = Decimal(1) / (1 << index)
            gain *= (1 + shift * shift).sqrt()
            if index:
                angles.append(_fixed(_arctan(shift), internal))
        return _CordicConstants(
            tuple(angles),
            _fixed(1 / gain, internal),
            _fixed(pi, internal),
            _fixed(pi / 2, internal),
            _fixed(pi, fraction + _RANGE_GUARD_BITS),
        )


def _check_function(function: str, data_width: int, fraction: int) -> None:
    """Validate the operation and the shared signed fixed-point geometry."""
    if not isinstance(function, str):
        raise TypeError("function must be a string")
    if function not in _FUNCTIONS:
        raise ValueError(f"{function!r} has no circular CORDIC unit")
    if type(data_width) is not int or type(fraction) is not int:
        raise TypeError("CORDIC geometry requires integer widths")
    check_format(data_width, fraction)


def cordic_latency(function: str, *, data_width: int = 16, fraction: int = 8) -> int:
    """Return the edges from accepting a request to asserting result valid.

    Parameters
    ----------
    function : str
        One of sin, cos, arctan or atan2.
    data_width, fraction : int, optional
        Signed input/output word geometry.

    Returns
    -------
    int
        Fixed latency in clock edges after the input handshake.

    Raises
    ------
    TypeError
        If the operation or geometry has the wrong type.
    ValueError
        If the operation or geometry is unsupported.

    """
    _check_function(function, data_width, fraction)
    internal = fraction + _GUARD_BITS
    preparation = (
        data_width + _RANGE_GUARD_BITS + _ROTATION_SETUP_EDGES
        if function in _ROTATION_FUNCTIONS
        else internal
    )
    return preparation + internal + 2


def _rotate(code: int, function: str, constants: _CordicConstants) -> int:
    """Evaluate the circular rotation and fold the result into its quadrant."""
    wide_pi = constants.wide_pi
    mapped = code << _RANGE_GUARD_BITS
    wide_z = (mapped + wide_pi) % (2 * wide_pi) - wide_pi
    z = shift_round(wide_z, _GUARD_BITS - _RANGE_GUARD_BITS)
    sign = 1
    if z > constants.half_pi:
        z -= constants.pi
        sign = -1
    elif z < -constants.half_pi:
        z += constants.pi
        sign = -1
    x, y = constants.inverse_gain, 0
    for index, angle in enumerate(constants.angles):
        direction = 1 if z >= 0 else -1
        x, y, z = (
            x - direction * (y >> index),
            y + direction * (x >> index),
            z - direction * angle,
        )
    return sign * (y if function == "sin" else x)


def _vector(y: int, x: int, internal: int, constants: _CordicConstants) -> int:
    """Evaluate atan2 with explicit axes, origin and shared vector scaling."""
    if y == 0:
        return constants.pi if x < 0 else 0
    if x == 0:
        return -constants.half_pi if y < 0 else constants.half_pi
    negative_x, negative_y = x < 0, y < 0
    x, y = abs(x), abs(y)
    shift = max(0, internal + 1 - max(x.bit_length(), y.bit_length()))
    x, y = x << shift, y << shift
    z = 0
    for index, angle in enumerate(constants.angles):
        if y == 0:
            break
        direction = 1 if y > 0 else -1
        x, y, z = (
            x + direction * (y >> index),
            y - direction * (x >> index),
            z + direction * angle,
        )
    if negative_x:
        z = constants.pi - z
    return -z if negative_y else z


def _input_code(code: int, width: int) -> None:
    """Validate one operand before applying the integer recurrence."""
    if type(code) is not int:
        raise TypeError("CORDIC operands must be signed integer codes")
    limit = 1 << (width - 1)
    if not -limit <= code < limit:
        raise ValueError(f"CORDIC operand outside the signed {width}-bit domain")


def evaluate_cordic(
    function: str,
    argument0: int,
    *,
    argument1: int | None = None,
    data_width: int = 16,
    fraction: int = 8,
) -> int:
    """Evaluate the clocked unit's bit-true fixed-point arithmetic.

    Parameters
    ----------
    function : str
        One of sin, cos, arctan or atan2.
    argument0 : int
        Signed input code, or the vertical coordinate for atan2.
    argument1 : int or None, optional
        Signed horizontal coordinate, required only for atan2.
    data_width, fraction : int, optional
        Signed input/output word geometry.

    Returns
    -------
    int
        The result code rounded half up and saturated to the signed word.

    Raises
    ------
    TypeError
        If an argument or the geometry has the wrong type.
    ValueError
        If the operation, operand count, code or geometry is unsupported.

    """
    _check_function(function, data_width, fraction)
    _input_code(argument0, data_width)
    if (function == "atan2") != (argument1 is not None):
        raise ValueError("Only atan2 requires a second operand")
    if argument1 is not None:
        _input_code(argument1, data_width)
    constants = _constants(fraction)
    if function in _ROTATION_FUNCTIONS:
        value = _rotate(argument0, function, constants)
    else:
        x = 1 << fraction if argument1 is None else argument1
        value = _vector(argument0, x, fraction + _GUARD_BITS, constants)
    return saturate(shift_round(value, -_GUARD_BITS), data_width)
