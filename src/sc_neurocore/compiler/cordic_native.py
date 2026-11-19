# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Portable C and Rust circular CORDIC emission

"""Emit the same integer recurrence for standalone native compiler consumers."""

from .cordic import (
    _GUARD_BITS,
    _RANGE_GUARD_BITS,
    _ROTATION_FUNCTIONS,
    _CordicConstants,
    _check_function,
    _constants,
)

_C_HEADER = """// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Portable circular CORDIC
"""


def _c_word(value: int) -> str:
    """Write a nonnegative integer as two 64-bit constant limbs."""
    return f"{{ UINT64_C({value >> 64}), UINT64_C({value & ((1 << 64) - 1)}) }}"


def _c_reduction(data_width: int, wide_pi: int) -> str:
    """Generate a C11 wide remainder without implementation-defined signed shifts."""
    shift = _RANGE_GUARD_BITS - _GUARD_BITS
    return f"""
typedef struct {{ uint64_t hi; uint64_t lo; }} cordic_word;
static bool cordic_ge(cordic_word a, cordic_word b) {{
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}}
static cordic_word cordic_subtract(cordic_word a, cordic_word b) {{
    cordic_word result = {{ a.hi - b.hi - (a.lo < b.lo), a.lo - b.lo }};
    return result;
}}
static int64_t cordic_phase(int32_t code) {{
    const cordic_word pi = {_c_word(wide_pi)};
    const cordic_word period = {_c_word(2 * wide_pi)};
    const uint64_t magnitude = code < 0 ? (uint64_t)(-(int64_t)code) : (uint64_t)code;
    cordic_word residue = {{0, 0}};
    for (unsigned n = {data_width + _RANGE_GUARD_BITS}; n > 0; --n) {{
        const unsigned index = n - 1;
        const uint64_t bit = index < {_RANGE_GUARD_BITS} ? 0 :
            (magnitude >> (index - {_RANGE_GUARD_BITS})) & UINT64_C(1);
        residue.hi = (residue.hi << 1) | (residue.lo >> 63);
        residue.lo = (residue.lo << 1) | bit;
        if (cordic_ge(residue, period)) residue = cordic_subtract(residue, period);
    }}
    const bool upper_half = cordic_ge(residue, pi);
    const bool exact_pi = residue.hi == pi.hi && residue.lo == pi.lo;
    const bool negative = exact_pi || (upper_half != (code < 0));
    if (upper_half) residue = cordic_subtract(period, residue);
    const uint64_t quotient = (residue.hi << {64 - shift}) | (residue.lo >> {shift});
    const uint64_t tail = residue.lo & UINT64_C({(1 << shift) - 1});
    const uint64_t half = UINT64_C({1 << (shift - 1)});
    if (negative) return -(int64_t)(quotient + (tail > half));
    return (int64_t)(quotient + (tail >= half));
}}
"""


def _c_rotation(function: str, constants: _CordicConstants) -> str:
    """Generate rotation over the same finite sequence of circular angles."""
    coordinate = "y" if function == "sin" else "x"
    return f"""
    int64_t z = cordic_phase(argument0);
    const bool opposite = z > {constants.half_pi} || z < -{constants.half_pi};
    if (z > {constants.half_pi}) z -= {constants.pi};
    else if (z < -{constants.half_pi}) z += {constants.pi};
    int64_t x = {constants.inverse_gain}, y = 0;
    for (unsigned i = 0; i < {len(constants.angles)}; ++i) {{
        const int64_t direction = z >= 0 ? 1 : -1;
        const int64_t next_x = x - direction * cordic_floor_shift(y, i);
        y += direction * cordic_floor_shift(x, i);
        x = next_x;
        z -= direction * angles[i];
    }}
    const int64_t value = opposite ? -{coordinate} : {coordinate};
"""


def _c_vector(function: str, fraction: int, constants: _CordicConstants) -> str:
    """Generate vectoring and the exact axial cases."""
    horizontal = "argument1" if function == "atan2" else str(1 << fraction)
    normalised = 1 << (fraction + _GUARD_BITS)
    return f"""
    int64_t x = {horizontal}, y = argument0, value;
    if (y == 0) value = x < 0 ? {constants.pi} : 0;
    else if (x == 0) value = y < 0 ? -{constants.half_pi} : {constants.half_pi};
    else {{
        const bool negative_x = x < 0, negative_y = y < 0;
        if (negative_x) x = -x;
        if (negative_y) y = -y;
        while (x < INT64_C({normalised}) && y < INT64_C({normalised})) {{
            x *= 2;
            y *= 2;
        }}
        int64_t z = 0;
        for (unsigned i = 0; i < {len(constants.angles)} && y != 0; ++i) {{
            const int64_t direction = y > 0 ? 1 : -1;
            const int64_t next_x = x + direction * cordic_floor_shift(y, i);
            y -= direction * cordic_floor_shift(x, i);
            x = next_x;
            z += direction * angles[i];
        }}
        if (negative_x) z = {constants.pi} - z;
        value = negative_y ? -z : z;
    }}
"""


def emit_cordic_c(function: str, *, data_width: int = 16, fraction: int = 8) -> str:
    """Emit portable C11 source for the clocked unit's integer result.

    Parameters
    ----------
    function : str
        One of sin, cos, arctan or atan2.
    data_width, fraction : int, optional
        Signed input and output geometry.

    Returns
    -------
    str
        Source exporting ``bool cordic_evaluate(int32_t argument0, ..., int32_t *result)``.
        Only atan2 takes argument1. False indicates invalid operands or a null result pointer.

    Raises
    ------
    TypeError
        If the operation or geometry has the wrong type.
    ValueError
        If the operation or geometry is unsupported.

    """
    _check_function(function, data_width, fraction)
    constants = _constants(fraction)
    limit = 1 << (data_width - 1)
    second = ", int32_t argument1" if function == "atan2" else ""
    if data_width == 32:
        reject = ""
    else:
        reject = f" || argument0 < -{limit} || argument0 >= {limit}"
        if second:
            reject += f" || argument1 < -{limit} || argument1 >= {limit}"
    rotation = function in _ROTATION_FUNCTIONS
    phase = _c_reduction(data_width, constants.wide_pi) if rotation else ""
    arithmetic = (
        _c_rotation(function, constants) if rotation else _c_vector(function, fraction, constants)
    )
    angles = ", ".join(map(str, constants.angles))
    return (
        _C_HEADER
        + f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

static const int64_t angles[] = {{ {angles} }};

static int64_t cordic_floor_shift(int64_t n, unsigned shift) {{
    return n < 0 ? -1 - ((-1 - n) >> shift) : n >> shift;
}}
{phase}
bool cordic_evaluate(int32_t argument0{second}, int32_t *result) {{
    if (result == NULL{reject}) return false;
{arithmetic}
    const int64_t rounded = cordic_floor_shift(value + {1 << (_GUARD_BITS - 1)}, {_GUARD_BITS});
    const int64_t clipped = rounded < -INT64_C({limit}) ? -INT64_C({limit}) :
                            rounded > INT64_C({limit - 1}) ? INT64_C({limit - 1}) : rounded;
    *result = (int32_t)clipped;
    return true;
}}
"""
    )


def _rust_rotation(function: str, constants: _CordicConstants) -> str:
    """Generate signed wide phase reduction and circular rotation in Rust."""
    coordinate = "y" if function == "sin" else "x"
    shift = _RANGE_GUARD_BITS - _GUARD_BITS
    return f"""
    let pi = {constants.wide_pi}i128;
    let wide = ((i128::from(argument0) << {_RANGE_GUARD_BITS}) + pi).rem_euclid(2 * pi) - pi;
    let mut z = ((wide + (1_i128 << {shift - 1})) >> {shift}) as i64;
    let opposite = z > {constants.half_pi} || z < -{constants.half_pi};
    if z > {constants.half_pi} {{ z -= {constants.pi}; }}
    else if z < -{constants.half_pi} {{ z += {constants.pi}; }}
    let (mut x, mut y) = ({constants.inverse_gain}_i64, 0_i64);
    for (i, angle) in ANGLES.iter().enumerate() {{
        let direction = if z >= 0 {{ 1 }} else {{ -1 }};
        (x, y) = (x - direction * (y >> i), y + direction * (x >> i));
        z -= direction * angle;
    }}
    let value = if opposite {{ -{coordinate} }} else {{ {coordinate} }};
"""


def _rust_vector(function: str, fraction: int, constants: _CordicConstants) -> str:
    """Generate the circular vectoring recurrence and full quadrant policy."""
    horizontal = "i64::from(argument1)" if function == "atan2" else f"{1 << fraction}_i64"
    normalised = 1 << (fraction + _GUARD_BITS)
    return f"""
    let (mut x, mut y) = ({horizontal}, i64::from(argument0));
    let value = if y == 0 {{
        if x < 0 {{ {constants.pi} }} else {{ 0 }}
    }} else if x == 0 {{
        if y < 0 {{ -{constants.half_pi} }} else {{ {constants.half_pi} }}
    }} else {{
        let (negative_x, negative_y) = (x < 0, y < 0);
        (x, y) = (x.abs(), y.abs());
        while x < {normalised} && y < {normalised} {{
            x <<= 1;
            y <<= 1;
        }}
        let mut z = 0;
        for (i, angle) in ANGLES.iter().enumerate() {{
            if y == 0 {{ break; }}
            let direction = if y > 0 {{ 1 }} else {{ -1 }};
            (x, y) = (x + direction * (y >> i), y - direction * (x >> i));
            z += direction * angle;
        }}
        if negative_x {{ z = {constants.pi} - z; }}
        if negative_y {{ -z }} else {{ z }}
    }};
"""


def emit_cordic_rust(function: str, *, data_width: int = 16, fraction: int = 8) -> str:
    """Emit standard-library Rust source for the clocked unit's integer result.

    Parameters
    ----------
    function : str
        One of sin, cos, arctan or atan2.
    data_width, fraction : int, optional
        Signed input and output geometry.

    Returns
    -------
    str
        Source exporting ``cordic_evaluate(argument0: i32, ...) -> Option<i32>``.
        Only atan2 takes argument1. None indicates an invalid operand.

    Raises
    ------
    TypeError
        If the operation or geometry has the wrong type.
    ValueError
        If the operation or geometry is unsupported.

    """
    _check_function(function, data_width, fraction)
    constants = _constants(fraction)
    limit = 1 << (data_width - 1)
    second = ", argument1: i32" if function == "atan2" else ""
    reject = ""
    if data_width != 32:
        reject = f"!(-{limit}..{limit}).contains(&argument0)"
        if second:
            reject += f" || !(-{limit}..{limit}).contains(&argument1)"
        reject = f"\n    if {reject} {{ return None; }}\n"
    arithmetic = (
        _rust_rotation(function, constants)
        if function in _ROTATION_FUNCTIONS
        else _rust_vector(function, fraction, constants)
    )
    angles = ", ".join(map(str, constants.angles))
    return (
        _C_HEADER
        + f"""
const ANGLES: [i64; {len(constants.angles)}] = [{angles}];

pub fn cordic_evaluate(argument0: i32{second}) -> Option<i32> {{{reject}
{arithmetic}
    let rounded = (value + {1 << (_GUARD_BITS - 1)}) >> {_GUARD_BITS};
    Some(rounded.clamp(-{limit}, {limit - 1}) as i32)
}}
"""
    )
