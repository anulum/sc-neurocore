# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand() and randn() as 64-bit integer programs over Philox4x32-10

"""Build ``rand``, ``randn`` and ``poisson`` units as :class:`~sc_neurocore.compiler.function_unit_ir.Program` values.

The Philox4x32-10 rounds of :mod:`sc_neurocore.compiler.random_units` are
written in the 64-bit integer program: each 32×32-bit product is formed from two
16-bit partial products (each below ``2**48``), so its high and low words are
exact without a 64-bit unsigned type. The programs read three named external
inputs: ``step`` (the step counter), ``seed`` (the 64-bit key, two's complement)
and ``stream`` (the instance number); the call site is a constant. The tests
evaluate the programs against :func:`~sc_neurocore.compiler.random_units.draw`,
:func:`~sc_neurocore.compiler.random_units.uniform_code`,
:func:`~sc_neurocore.compiler.random_units.normal_code` and
:func:`~sc_neurocore.compiler.random_units.poisson_code`. A ``poisson`` unit
compares word 0 with the constant thresholds of its mean, so the mean is fixed
when the unit is built. A ``poisson_grid`` unit takes its mean as the program
input (a code of the output format), floors it to the 1/16 grid on ``[0, 16]``
and compares word 0 with that row of
:func:`~sc_neurocore.compiler.random_units.poisson_grid_table`, packed three
thresholds to a table row.
"""

from __future__ import annotations

from functools import cache

from . import exact_normal as xn
from . import function_units as fu
from . import random_units as ru
from .function_unit_ir import Program
from .function_unit_programs import (
    _abs,
    _isqrt,
    _log2_internal,
    _saturate,
    _shift_round,
    _turn_sine,
)

_MASK = (1 << 32) - 1
KINDS = ("rand", "randn", "randn_exact", "poisson", "poisson_grid")
EXTERNS = ("step", "seed", "stream")


def _mulhilo(p: Program, word: str, multiplier: int) -> tuple[str, str]:
    """High and low 32-bit words of ``word · multiplier`` from 16-bit partial products."""
    low_part = p.mul(word, p.const(multiplier & 0xFFFF))
    high_part = p.mul(word, p.const(multiplier >> 16))
    high = p.shr(p.add(high_part, p.shr(low_part, 16)), 16)
    low = p.mask(p.add(p.shl(p.mask(high_part, 16), 16), low_part), 32)
    return high, low


def _philox(p: Program, site: int) -> tuple[str, str, str, str]:
    step = p.extern("step")
    seed = p.extern("seed")
    c0 = p.const(site)
    c1 = p.mask(step, 32)
    c2 = p.mask(p.shr(step, 32), 32)
    c3 = p.mask(p.extern("stream"), 32)
    k0 = p.mask(seed, 32)
    k1 = p.mask(p.shr(seed, 32), 32)
    for round_index in range(ru.ROUNDS):
        if round_index:
            k0 = p.mask(p.add(k0, p.const(ru.WEYL_0)), 32)
            k1 = p.mask(p.add(k1, p.const(ru.WEYL_1)), 32)
        high0, low0 = _mulhilo(p, c0, ru.MULTIPLIER_0)
        high1, low1 = _mulhilo(p, c2, ru.MULTIPLIER_1)
        c0, c1, c2, c3 = p.xor(p.xor(high1, c1), k0), low1, p.xor(p.xor(high0, c3), k1), low0
    return c0, c1, c2, c3


def _rand(p: Program, site: int, data_width: int, fraction: int) -> str:
    word0 = _philox(p, site)[0]
    return p.shr(word0, 32 - fraction)


def _randn(p: Program, site: int, data_width: int, fraction: int) -> str:
    word0, word1, _, _ = _philox(p, site)
    log2u = _log2_internal(p, p.add(word0, p.const(1)), 32)
    whole = p.shr(log2u, fu.P)
    rest = p.sub(log2u, p.shl(whole, fu.P))
    natural = p.add(p.mul(whole, p.const(fu._LN2)), p.shr(p.mul(rest, p.const(fu._LN2)), fu.P))
    square = p.mul(p.const(-2), natural)
    # Series radius near u = 1, as ru.normal_code computes it. Both sides are always
    # computed, so a far word's distance is replaced by 0 before it is squared.
    full_distance = p.sub(p.const((1 << 32) - 1), word0)
    is_near = p.lt(full_distance, p.const(ru.NEAR_ONE))
    distance = p.select(is_near, full_distance, p.const(0))
    square_d = p.mul(distance, distance)
    near = p.add(
        p.add(p.shl(distance, 28), p.shr(square_d, 5)),
        p.div(p.mul(p.shr(square_d, 18), distance), p.const(3 << 18)),
    )
    radius = _isqrt(p, p.select(is_near, p.shr(near, 3), p.shl(square, 26)))
    turn = p.shr(word1, 2)
    cosine = _turn_sine(p, p.mask(p.add(turn, p.const(1 << (fu.B - 2))), fu.B))
    return _saturate(p, _shift_round(p, p.mul(radius, cosine), fraction - 58), data_width)


def _mul60(p: Program, a: str, b: str) -> str:
    """``(a·b) >> 60`` of two non-negative values below ``2**62``, as :func:`exact_normal.mul60`."""
    a1, a0 = p.shr(a, 31), p.mask(a, 31)
    b1, b0 = p.shr(b, 31), p.mask(b, 31)
    cross = p.add(p.shr(p.mul(a1, b0), 1), p.shr(p.mul(a0, b1), 1))
    return p.add(p.shl(p.mul(a1, b1), 2), p.shr(p.add(cross, p.shr(p.mul(a0, b0), 32)), 28))


def _top_bit(p: Program, value: str) -> str:
    """Index of the highest set bit of a positive value below ``2**63`` (0 for 0), by halving."""
    position, rest = p.const(0), value
    for step in (32, 16, 8, 4, 2, 1):
        shifted = p.shr(rest, step)
        above = p.lt(p.const(0), shifted)
        rest = p.select(above, shifted, rest)
        position = p.add(position, p.select(above, p.const(step), p.const(0)))
    return position


def _shift_signed(p: Program, value: str, amount: str) -> str:
    """``value << amount`` for a non-negative amount, ``value >> -amount`` otherwise.

    Both shifts are formed with the other direction's amount clamped to 0, so the
    branch that is not selected never shifts a large value left.
    """
    zero = p.const(0)
    negative = p.lt(amount, zero)
    left = p.select(negative, zero, amount)
    right = p.select(negative, p.neg(amount), zero)
    return p.shr_var(p.shl_var(value, left), right)


def _randn_exact(p: Program, site: int, data_width: int, fraction: int) -> str:
    word0, word1, _, _ = _philox(p, site)
    one = p.const(xn.ONE)
    zero = p.const(0)
    ln2_58, two_pi_38 = xn.tables()["const"]
    # -ln u near u = 1: the series in x = 1 - u (s at fraction 59).
    distance = p.sub(p.const((1 << 32) - 1), word0)
    x = p.shl(distance, 28)
    acc = p.const(xn.ONE // 6)
    for divisor in (5, 4, 3, 2):
        acc = p.add(p.const(xn.ONE // divisor), _mul60(p, x, acc))
    near = _mul60(p, x, p.add(one, _mul60(p, x, acc)))
    # -ln u elsewhere: mantissa, table point, ln(1 + r) to r^5/5 (s at fraction 57).
    n = p.add(word0, p.const(1))
    e_n = _top_bit(p, n)
    m = p.shl_var(n, p.sub(p.const(xn.F), e_n))
    point = p.mask(p.shr(m, xn.F - 8), 8)
    r = p.sub(_mul60(p, m, p.lookup("xn_log", point, 1)), one)
    r = p.select(p.lt(r, zero), zero, r)
    acc = p.const(xn.ONE // 5)
    for divisor in (4, 3, 2, 1):
        acc = p.sub(p.const(xn.ONE // divisor), _mul60(p, r, acc))
    ln_m = p.add(p.lookup("xn_log", point, 0), _mul60(p, r, acc))
    far = p.sub(p.mul(p.sub(p.const(32), e_n), p.const(ln2_58)), p.shr(ln_m, 2))
    far = p.select(p.lt(far, zero), zero, far)
    is_near = p.lt(distance, p.const(xn.NEAR_ONE_WORDS))
    s = p.select(is_near, near, far)
    s_fraction = p.select(is_near, p.const(59), p.const(57))
    # sqrt(s) at fraction 58: s = 4**e m, m in [1, 4); three inverse-square-root Newton steps.
    e = p.shr(p.sub(_top_bit(p, s), s_fraction), 1)
    mantissa = _shift_signed(p, s, p.sub(p.sub(p.const(xn.F), s_fraction), p.shl(e, 1)))
    index = p.sub(p.shr(mantissa, 54), p.const(64))
    y = p.lookup("xn_rsqrt", p.select(p.lt(index, zero), zero, index), 0)
    three = p.const(3 * xn.ONE)
    for _ in range(3):
        y = p.shr(_mul60(p, y, p.sub(three, _mul60(p, mantissa, _mul60(p, y, y)))), 1)
    root = _mul60(p, mantissa, y)
    radius = _shift_signed(p, root, p.sub(e, p.const(2)))
    radius = p.select(p.eq(s, zero), zero, radius)
    # cos(2π word1 / 2**32): quadrant, 256 segments, Taylor remainder.
    quadrant = p.shr(word1, 30)
    within = p.mask(word1, 30)
    segment = p.shr(within, 22)
    b = p.shr(p.mul(p.mask(within, 22), p.const(two_pi_38)), 10)
    b2 = _mul60(p, b, b)
    b3 = _mul60(p, b, b2)
    cos_b = p.add(p.sub(one, p.shr(b2, 1)), p.div(_mul60(p, b2, b2), p.const(24)))
    sin_b = p.add(p.sub(b, p.div(b3, p.const(6))), p.div(_mul60(p, b3, b2), p.const(120)))
    cos_k, sin_k = p.lookup("xn_trig", segment, 0), p.lookup("xn_trig", segment, 1)
    cos = p.sub(_mul60(p, cos_k, cos_b), _mul60(p, sin_k, sin_b))
    sin = p.add(_mul60(p, sin_k, cos_b), _mul60(p, cos_k, sin_b))
    turn_cos = p.select(
        p.eq(quadrant, zero),
        cos,
        p.select(
            p.eq(quadrant, p.const(1)),
            p.neg(sin),
            p.select(p.eq(quadrant, p.const(2)), p.neg(cos), sin),
        ),
    )
    # One rounding to the output fraction, ties away from zero, then saturation.
    magnitude = _mul60(p, radius, _abs(p, turn_cos))
    shift = 58 - fraction
    rounded = p.shr(p.add(magnitude, p.const(1 << (shift - 1))), shift)
    value = p.select(p.lt(turn_cos, zero), p.neg(rounded), rounded)
    return _saturate(p, value, data_width)


def _exact_tables() -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Return the exact mode's tables as rows of three coefficients."""
    t = xn.tables()
    return {
        "xn_log": tuple(zip(t["ln"], t["recip"], (0,) * 256, strict=True)),
        "xn_trig": tuple(zip(t["cos"], t["sin"], (0,) * 256, strict=True)),
        "xn_rsqrt": tuple((y, 0, 0) for y in t["rsqrt"]),
    }


def _grid_tables() -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Return the grid thresholds as tables of triples: table ``j`` holds entries 3j..3j+2."""
    rows = ru.poisson_grid_table()
    width = len(rows[0])
    tables: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for j in range((width + 2) // 3):
        triples = []
        for row in rows:
            padded = row + (1 << 32,) * (3 * j + 3 - len(row))
            triples.append((padded[3 * j], padded[3 * j + 1], padded[3 * j + 2]))
        tables[f"poisson_grid_{j}"] = tuple(triples)
    return tables


def _poisson_grid(p: Program, site: int, data_width: int, fraction: int) -> str:
    word0 = _philox(p, site)[0]
    code = p.input()
    shift = fraction - 4
    raw = p.shr(code, shift) if shift >= 0 else p.shl(code, -shift)
    # Clamp both ends here: a Verilog case of a negative index would take the default row.
    last = p.const(ru.GRID_ROWS - 1)
    zero = p.const(0)
    index = p.select(p.lt(raw, zero), zero, p.select(p.lt(last, raw), last, raw))
    count = p.const(0)
    width = len(ru.poisson_grid_table()[0])
    for k in range(width):
        threshold = p.lookup(f"poisson_grid_{k // 3}", index, k % 3)
        count = p.add(count, p.le(threshold, word0))
    return _saturate(p, p.shl(count, fraction), data_width)


def _poisson(p: Program, site: int, data_width: int, fraction: int, lam: float) -> str:
    word0 = _philox(p, site)[0]
    count = p.const(0)
    for threshold in ru.poisson_thresholds(lam):
        count = p.add(count, p.le(p.const(threshold), word0))
    return _saturate(p, p.shl(count, fraction), data_width)


@cache
def program(
    kind: str, site: int, data_width: int, fraction: int, lam: float | None = None
) -> Program:
    """Return the ``rand``, ``randn`` or ``poisson`` unit for one call site and format.

    Parameters
    ----------
    kind : str
        ``"rand"``, ``"randn"``, ``"poisson"`` or ``"poisson_grid"`` (whose mean is
        the program input).
    site : int
        Call-site number, ``[0, 2**32)``.
    data_width, fraction : int
        The output format; ``rand`` needs ``fraction <= 32``.
    lam : float, optional
        The mean of a ``poisson`` unit, ``[0, MAX_POISSON_LAMBDA]``; required for
        ``poisson`` and refused for the others.

    Returns
    -------
    Program
        A program with external inputs ``step``, ``seed`` and ``stream``.

    Raises
    ------
    ValueError
        For an unknown kind, a site outside 32 bits, a format the units cannot mirror,
        or a missing, superfluous or out-of-range mean.

    """
    if kind not in KINDS:
        raise ValueError(f"no random unit {kind!r}")
    if not 0 <= site <= _MASK:
        raise ValueError(f"call site {site} is outside 32 bits")
    fu.check_format(data_width, fraction)
    if (kind == "poisson") != (lam is not None):
        raise ValueError("a mean is given to, and only to, a poisson unit")
    if kind == "poisson_grid":
        p = Program(tables=_grid_tables())
        p.output = _poisson_grid(p, site, data_width, fraction)
        return p
    if lam is not None:
        ru.poisson_thresholds(lam)  # range check before any step is built
        p = Program(tables={})
        p.output = _poisson(p, site, data_width, fraction, lam)
        return p
    if kind == "randn_exact":
        # check_format limits words to 32 bits, so the fraction is below 58, where
        # the unit rounds from.
        p = Program(tables=_exact_tables())
        p.output = _randn_exact(p, site, data_width, fraction)
        return p
    p = Program(tables={name: fu.table(name) for name in fu.SEGMENT_BITS})
    build = _rand if kind == "rand" else _randn
    p.output = build(p, site, data_width, fraction)
    return p


__all__ = ["EXTERNS", "KINDS", "program"]
