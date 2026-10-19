# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — randn() at 60-bit internal precision: the integer reference of the exact mode

"""``randn()`` with 60-bit internals: the integer reference of ``random_precision="exact"``.

The standard ``randn`` unit (:func:`sc_neurocore.compiler.random_units.normal_code`)
computes the Box–Muller transform with 30-bit internals; its code is within half an
LSB plus ``2**-24`` of the model's value, which at a high fraction is many LSB.
This mode computes the same transform of the same two Philox words,
``sqrt(-2 ln((w0 + 1)/2**32)) · cos(2π w1 / 2**32)``, with every intermediate
held to about 60 bits, and rounds once to the output word, so the code is within
half an LSB plus :data:`EXACT_NORMAL_TRANSFORM_ERROR` (``2**-45``; measured largest
``1.4e-14`` ≈ ``2**-46``) of the exact transform: at a 30-bit fraction that is
0.5 LSB plus 1/32768 LSB.

Every step stays inside a signed 64-bit integer, so the same arithmetic is a
straight-line program for Verilog, C and Rust
(:mod:`sc_neurocore.compiler.random_unit_programs`):

* :func:`mul60` multiplies two non-negative values below ``2**62`` and returns
  ``(a·b) >> 60`` from 31-bit limbs, every partial product below ``2**62``;
* ``-ln u``: for ``u`` within ``2**-8`` of 1 the series ``x + x²/2 + … + x⁶/6`` in
  ``x = 1 - u`` (fraction 60); otherwise ``u = m·2**(e-32)`` with ``m`` in ``[1, 2)``,
  ``ln m = ln c_k + ln(1 + r)`` for the 256 table points ``c_k = 1 + k/256``,
  ``r = m/c_k - 1 < 2**-8`` (a reciprocal table) and ``ln(1 + r)`` to ``r⁵/5``;
* ``sqrt(s)``: ``s = 4**e · m`` with ``m`` in ``[1, 4)``, three Newton steps of the
  inverse square root from a 192-entry table, then ``m · y``;
* ``cos``: the 32-bit turn splits into a quadrant, 256 segments (cos/sin table) and
  a remainder below ``2π·2**-10`` with ``cos b = 1 - b²/2 + b⁴/24`` and
  ``sin b = b - b³/6 + b⁵/120``.

The tables are computed once in decimal arithmetic at 60 digits.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import cache

__all__ = [
    "EXACT_NORMAL_TRANSFORM_ERROR",
    "F",
    "ONE",
    "mul60",
    "neg_log_u",
    "normal_code_exact",
    "sqrt_fraction58",
    "tables",
    "turn_cosine",
]

F = 60
"""Internal fraction bits."""
ONE = 1 << F
"""1.0 at the internal fraction."""

EXACT_NORMAL_TRANSFORM_ERROR = 2.0**-45
"""Bound on |transform before the final rounding - exact transform|; measured 1.4e-14."""

NEAR_ONE_WORDS = 1 << 24
"""``u`` within ``2**-8`` of 1 (fewer than this many words below ``2**32``) takes the series."""

_LIMB = (1 << 31) - 1


def mul60(a: int, b: int) -> int:
    """Return ``(a·b) >> 60`` for ``0 <= a, b < 2**62`` with every partial product below ``2**62``.

    Parameters
    ----------
    a, b : int
        Non-negative operands at fraction 60 (or any fractions whose sum is 60
        more than the result's).

    Returns
    -------
    int
        The product shifted right by 60, truncated (the partial shifts lose at
        most two units of the result).
    """
    a1, a0 = a >> 31, a & _LIMB
    b1, b0 = b >> 31, b & _LIMB
    cross = ((a1 * b0) >> 1) + ((a0 * b1) >> 1)
    return ((a1 * b1) << 2) + ((cross + ((a0 * b0) >> 32)) >> 28)


def _cos_sin(x: Decimal) -> tuple[Decimal, Decimal]:
    """Return cos and sin of ``x`` (|x| <= π/2) by their Taylor series in decimal arithmetic."""
    cos, sin, term = Decimal(1), x, Decimal(1)
    sin_term = x
    for n in range(1, 40):
        term = -term * x * x / ((2 * n - 1) * (2 * n))
        sin_term = -sin_term * x * x / ((2 * n) * (2 * n + 1))
        cos += term
        sin += sin_term
    return cos, sin


@cache
def tables() -> dict[str, tuple[int, ...]]:
    """Return the tables of the exact mode, at fraction 60.

    Returns
    -------
    dict
        ``ln``: ``ln(1 + k/256)``; ``recip``: ``1/(1 + k/256)``; ``cos``/``sin``:
        of ``k·(π/2)/256``; ``rsqrt``: ``1/sqrt(1 + (k + 1/2)/64)`` for the 192
        points of ``[1, 4)``; ``const``: ``ln 2`` at fraction 58 and ``2π·2**38``.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        one = Decimal(ONE)
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
        points = [1 + Decimal(k) / 256 for k in range(256)]
        cos_sin = [_cos_sin(pi / 2 * k / 256) for k in range(256)]
        return {
            "ln": tuple(int((c.ln() * one).to_integral_value()) for c in points),
            "recip": tuple(int((one / c).to_integral_value()) for c in points),
            "cos": tuple(int((c * one).to_integral_value()) for c, _ in cos_sin),
            "sin": tuple(int((s * one).to_integral_value()) for _, s in cos_sin),
            "rsqrt": tuple(
                int((one / (1 + (Decimal(k) + Decimal("0.5")) / 64).sqrt()).to_integral_value())
                for k in range(192)
            ),
            "const": (
                int((Decimal(2).ln() * (1 << 58)).to_integral_value()),
                int((2 * pi * (1 << 38)).to_integral_value()),
            ),
        }


def neg_log_u(word0: int) -> tuple[int, int]:
    """Return ``-ln((word0 + 1)/2**32)`` and the fraction of ``s = -2 ln u`` it represents.

    Parameters
    ----------
    word0 : int
        Philox word 0.

    Returns
    -------
    tuple of int
        ``(L, fs)``: ``L`` is ``-ln u`` at fraction ``fs + 1``, which is ``s`` at
        fraction ``fs`` — 59 for the near-one series, 57 otherwise (so that ``L``
        up to 22.2 still fits 63 bits).
    """
    t = tables()
    distance = (1 << 32) - 1 - word0
    if distance < NEAR_ONE_WORDS:
        x = distance << 28  # 1 - u at fraction 60
        acc = ONE // 6
        for k in (5, 4, 3, 2):
            acc = ONE // k + mul60(x, acc)
        acc = ONE + mul60(x, acc)
        return mul60(x, acc), 59
    n = word0 + 1
    e = n.bit_length() - 1
    m = n << (F - e)  # [1, 2) at fraction 60
    k = (m >> (F - 8)) & 0xFF
    r = max(mul60(m, t["recip"][k]) - ONE, 0)
    acc = ONE // 5
    for divisor in (4, 3, 2, 1):
        acc = ONE // divisor - mul60(r, acc)
    ln_m = t["ln"][k] + mul60(r, acc)
    return max((32 - e) * t["const"][0] - (ln_m >> 2), 0), 57


def sqrt_fraction58(s: int, fraction: int) -> int:
    """Return ``sqrt(s / 2**fraction)`` at fraction 58 by inverse-square-root Newton steps.

    Parameters
    ----------
    s : int
        Non-negative input word.
    fraction : int
        Its fraction (57 or 59).

    Returns
    -------
    int
        The square root at fraction 58 (0 for 0).
    """
    if s == 0:
        return 0
    e = (s.bit_length() - 1 - fraction) // 2  # s / 4**e in [1, 4)
    shift = F - fraction - 2 * e
    m = s << shift if shift >= 0 else s >> -shift
    y = tables()["rsqrt"][(m >> 54) - 64]
    for _ in range(3):
        y = mul60(y, 3 * ONE - mul60(m, mul60(y, y))) >> 1
    root = mul60(m, y)  # sqrt(m) at fraction 60
    return root << e >> 2 if e >= 0 else root >> (2 - e)


def turn_cosine(word1: int) -> int:
    """Return ``cos(2π·word1/2**32)`` at fraction 60 (signed).

    Parameters
    ----------
    word1 : int
        Philox word 1, a fraction of a turn.

    Returns
    -------
    int
        The cosine.
    """
    t = tables()
    quadrant, within = word1 >> 30, word1 & ((1 << 30) - 1)
    k, low = within >> 22, within & ((1 << 22) - 1)
    b = (low * t["const"][1]) >> 10  # remainder angle at fraction 60, < 2π·2**-10
    b2 = mul60(b, b)
    b3 = mul60(b, b2)
    cos_b = ONE - (b2 >> 1) + mul60(b2, b2) // 24
    sin_b = b - b3 // 6 + mul60(b3, b2) // 120
    cos = mul60(t["cos"][k], cos_b) - mul60(t["sin"][k], sin_b)
    sin = mul60(t["sin"][k], cos_b) + mul60(t["cos"][k], sin_b)
    return (cos, -sin, -cos, sin)[quadrant]


def normal_code_exact(word0: int, word1: int, data_width: int, fraction: int) -> int:
    """Return the exact-mode code of ``randn()``: one rounding of the 60-bit transform.

    Parameters
    ----------
    word0, word1 : int
        Two Philox words.
    data_width, fraction : int
        The output format; ``fraction <= 57``.

    Returns
    -------
    int
        The code, rounded to nearest (ties away from zero) and saturated.
    """
    radius = sqrt_fraction58(*neg_log_u(word0))
    cos = turn_cosine(word1)
    magnitude = mul60(radius, abs(cos))  # fraction 58
    shift = 58 - fraction
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    value = rounded if cos >= 0 else -rounded
    limit = 1 << (data_width - 1)
    return max(-limit, min(limit - 1, value))
