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
integer normal must stay within half an LSB of the floating-point Box–Muller value
the Python model returns, up to the measured transform error
``NORMAL_TRANSFORM_ERROR``. Distribution checks state their sample size.
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


def _near_one_words() -> list[tuple[int, int]]:
    """Every word0 within 20 000 of 2**32, and the series boundary, with three phases."""
    distances = [*range(20000), ru.NEAR_ONE - 1, ru.NEAR_ONE, ru.NEAR_ONE + 1]
    return [((1 << 32) - 1 - d, w1) for d in distances for w1 in (0, (1 << 30) + 7, 1 << 31)]


def test_integer_normal_transform_error_is_below_its_bound() -> None:
    """At a 40-bit fraction the integer Box–Muller is within TRANSFORM_ERROR of the float one.

    Words near 2**32 (u close to 1, where the radius is steep) and 4000 drawn
    words; a measurement over these words, not a proof for all 2**64 pairs.
    """
    pairs = _near_one_words() + [ru.draw(1, s, 0, 2026)[:2] for s in range(4000)]
    worst = max(
        abs(ru.normal_code(w0, w1, 62, 40) / 2**40 - ru.normal_value(w0, w1)) for w0, w1 in pairs
    )
    assert worst <= ru.NORMAL_TRANSFORM_ERROR


def test_integer_normal_at_q16_16_near_u_one() -> None:
    """At Q16.16 every code is within half an LSB plus the transform error of the model.

    Before the series radius, word0 = 2**32 - 6 with word1 = 0 was 1.84 LSB away.
    """
    bound = 0.5 + ru.NORMAL_TRANSFORM_ERROR * 65536
    for w0, w1 in _near_one_words():
        model = max(-32768.0, min(32767.0, ru.normal_value(w0, w1)))
        assert abs(ru.normal_code(w0, w1, 32, 16) - model * 65536) <= bound, (w0, w1)


def test_program_takes_the_series_radius_where_the_reference_does() -> None:
    """Steps whose word0 lies within NEAR_ONE of 2**32: the program equals the reference."""
    near_steps = [s for s in range(6000) if (1 << 32) - 1 - ru.draw(1, s, 0, 2026)[0] < ru.NEAR_ONE]
    assert len(near_steps) >= 5
    for step in near_steps:
        words = ru.draw(1, step, 0, 2026)
        bound = {"step": step, "seed": 2026, "stream": 0}
        for dw, fr in ((16, 8), (32, 16), (32, 29)):
            assert program("randn", 1, dw, fr).evaluate(0, bound) == ru.normal_code(
                words[0], words[1], dw, fr
            )


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
