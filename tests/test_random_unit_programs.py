# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — rand()/randn() integer programs and their distributions

"""Tests of :mod:`sc_neurocore.compiler.random_unit_programs` and the draw helpers of ``random_units``.

The programs (emitted to Verilog, C and Rust) must equal the reference
:func:`~sc_neurocore.compiler.random_units.draw` with
:func:`~sc_neurocore.compiler.random_units.uniform_code` and
:func:`~sc_neurocore.compiler.random_units.normal_code`, including at the
32-bit boundaries of the step counter and with a seed above ``2**63``; the
integer normal must stay within 0.5 LSB of the floating-point Box–Muller value
the Python model returns. Distribution checks state their sample size.
"""

from __future__ import annotations

import math
import statistics

import pytest
import sympy

from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.random_unit_programs import program


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


@pytest.mark.parametrize("seed", [0, 12345, (1 << 64) - 1, 0xDEADBEEFCAFEF00D])
@pytest.mark.parametrize("step", [0, 1, (1 << 32) - 1, (1 << 32) + 5, (1 << 63) - 1])
def test_programs_equal_the_reference(seed: int, step: int) -> None:
    """Both programs equal the reference draw at two sites and two formats."""
    for site in (0, 7):
        words = ru.draw(site, step, 3, seed)
        bound = {"step": step, "seed": _signed(seed), "stream": 3}
        for dw, fr in ((16, 8), (32, 16)):
            assert program("rand", site, dw, fr).evaluate(0, bound) == ru.uniform_code(words[0], fr)
            assert program("randn", site, dw, fr).evaluate(0, bound) == ru.normal_code(
                words[0], words[1], dw, fr
            )


@pytest.mark.parametrize("lam", [0.0, 1e-9, 0.5, 3.5, 20.0, 100.0])
def test_poisson_program_equals_the_reference(lam: float) -> None:
    """The poisson program equals the reference inversion over 300 draws at two formats."""
    for step in range(300):
        words = ru.draw(5, step, 1, 77)
        bound = {"step": step, "seed": 77, "stream": 1}
        for dw, fr in ((16, 8), (32, 16)):
            got = program("poisson", 5, dw, fr, lam).evaluate(0, bound)
            assert got == ru.poisson_code(words[0], lam, dw, fr)


@pytest.mark.parametrize("lam", [0.25, 3.5, 47.0])
def test_poisson_thresholds_against_an_independent_sum(lam: float) -> None:
    """The thresholds equal ceil(2**32·CDF) from SymPy's exact rational sum at 60 digits."""
    rate = sympy.Rational(lam)
    partial = sympy.Integer(0)
    expected = []
    k = 0
    while True:
        partial += rate**k / sympy.factorial(k)
        threshold = int(sympy.ceiling((sympy.exp(-rate) * partial * 2**32).evalf(60)))
        if threshold >= 1 << 32:
            break
        expected.append(threshold)
        k += 1
    assert ru.poisson_thresholds(lam) == tuple(expected)


def test_poisson_distribution() -> None:
    """20 000 draws at mean 3.5: sample mean and variance within 0.05 and 0.1 of 3.5."""
    counts = [ru.poisson_count(ru.draw(2, s, 0, 9)[0], 3.5) for s in range(20000)]
    assert abs(statistics.mean(counts) - 3.5) < 0.05
    assert abs(statistics.pvariance(counts) - 3.5) < 0.1


def test_poisson_count_by_inversion_at_the_word_ends() -> None:
    """Word 0 gives the mode below the first threshold; the largest word gives the last count."""
    thresholds = ru.poisson_thresholds(3.5)
    assert ru.poisson_count(0, 3.5) == 0
    assert ru.poisson_count(thresholds[0], 3.5) == 1
    assert ru.poisson_count(thresholds[0] - 1, 3.5) == 0
    assert ru.poisson_count((1 << 32) - 1, 3.5) == len(thresholds)


def test_integer_normal_is_within_half_an_lsb_of_the_model() -> None:
    """Over 4000 draws at Q16.16 the integer Box–Muller equals the float one to 0.5 LSB."""
    worst = 0.0
    for step in range(4000):
        w = ru.draw(1, step, 0, 2026)
        model = max(-32768.0, min(32767.0, ru.normal_value(w[0], w[1])))
        worst = max(worst, abs(ru.normal_code(w[0], w[1], 32, 16) / 65536 - model) * 65536)
    assert worst <= 0.5


def test_distributions() -> None:
    """20 000 draws: uniform mean 0.5 ± 0.01; normal mean 0 ± 0.03 and standard deviation 1 ± 0.03."""
    uniform = [ru.uniform_value(ru.draw(0, s, 0, 7)[0]) for s in range(20000)]
    normal = [ru.normal_value(*ru.draw(1, s, 0, 7)[:2]) for s in range(20000)]
    assert abs(statistics.mean(uniform) - 0.5) < 0.01
    assert all(0.0 <= u < 1.0 for u in uniform)
    assert abs(statistics.mean(normal)) < 0.03
    assert abs(statistics.pstdev(normal) - 1.0) < 0.03


def test_uniform_code_is_the_floor_of_the_value() -> None:
    """The fixed-point uniform is ``floor(rand·2**fraction)`` at every fraction."""
    for word in (0, 1, 0x80000000, 0xFFFFFFFF):
        for fraction in (0, 8, 16, 30):
            assert ru.uniform_code(word, fraction) == math.floor(
                ru.uniform_value(word) * (1 << fraction)
            )


def test_refusals() -> None:
    """Unknown kinds, sites beyond 32 bits, unsupported formats and misplaced or bad means are refused."""
    with pytest.raises(ValueError, match="no random unit"):
        program("gamma", 0, 16, 8)
    with pytest.raises(ValueError, match="outside 32 bits"):
        program("rand", 1 << 32, 16, 8)
    with pytest.raises(ValueError):
        program("rand", 0, 64, 32)
    with pytest.raises(ValueError, match="only to, a poisson unit"):
        program("poisson", 0, 16, 8)
    with pytest.raises(ValueError, match="only to, a poisson unit"):
        program("rand", 0, 16, 8, 1.0)
    for bad in (-0.5, 100.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mean must be in"):
            program("poisson", 0, 16, 8, bad)
