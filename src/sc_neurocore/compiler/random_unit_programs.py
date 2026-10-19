# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand() and randn() as 64-bit integer programs over Philox4x32-10

"""Build ``rand`` and ``randn`` units as :class:`~sc_neurocore.compiler.function_unit_ir.Program` values.

The Philox4x32-10 rounds of :mod:`sc_neurocore.compiler.random_units` are
written in the 64-bit integer program: each 32×32-bit product is formed from two
16-bit partial products (each below ``2**48``), so its high and low words are
exact without a 64-bit unsigned type. The programs read three named external
inputs: ``step`` (the step counter), ``seed`` (the 64-bit key, two's complement)
and ``stream`` (the instance number); the call site is a constant. The tests
evaluate the programs against :func:`~sc_neurocore.compiler.random_units.draw`,
:func:`~sc_neurocore.compiler.random_units.uniform_code` and
:func:`~sc_neurocore.compiler.random_units.normal_code`.
"""

from __future__ import annotations

from functools import cache

from . import function_units as fu
from . import random_units as ru
from .function_unit_ir import Program
from .function_unit_programs import (
    _isqrt,
    _log2_internal,
    _saturate,
    _shift_round,
    _turn_sine,
)

_MASK = (1 << 32) - 1
KINDS = ("rand", "randn")
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
    radius = _isqrt(p, p.shl(square, 26))
    turn = p.shr(word1, 2)
    cosine = _turn_sine(p, p.mask(p.add(turn, p.const(1 << (fu.B - 2))), fu.B))
    return _saturate(p, _shift_round(p, p.mul(radius, cosine), fraction - 58), data_width)


@cache
def program(kind: str, site: int, data_width: int, fraction: int) -> Program:
    """Return the ``rand`` or ``randn`` unit for one call site and format.

    Parameters
    ----------
    kind : str
        ``"rand"`` or ``"randn"``.
    site : int
        Call-site number, ``[0, 2**32)``.
    data_width, fraction : int
        The output format; ``rand`` needs ``fraction <= 32``.

    Returns
    -------
    Program
        A program with external inputs ``step``, ``seed`` and ``stream``.

    Raises
    ------
    ValueError
        For an unknown kind, a site outside 32 bits or a format the units cannot mirror.

    """
    if kind not in KINDS:
        raise ValueError(f"no random unit {kind!r}")
    if not 0 <= site <= _MASK:
        raise ValueError(f"call site {site} is outside 32 bits")
    fu.check_format(data_width, fraction)
    p = Program(tables={name: fu.table(name) for name in fu.SEGMENT_BITS})
    build = _rand if kind == "rand" else _randn
    p.output = build(p, site, data_width, fraction)
    return p


__all__ = ["EXTERNS", "KINDS", "program"]
