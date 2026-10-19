# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — counter-based random numbers for compiled equations: the integer reference

"""Philox4x32-10, the counter-based generator the compiled random functions will draw from.

A draw is a pure function of a 128-bit counter and a 64-bit key (Salmon, Moraes,
Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC '11), so a
datapath needs no random state and a folded population no random-state memory;
repeating a step repeats its draws. The round and key schedule follow the
Random123 definition as distributed in ``randomgen`` (``src/philox/philox.h``):
each of the ten rounds multiplies words 0 and 2 by ``0xD2511F53`` and
``0xCD9E8D57`` into 64-bit products and mixes their high halves with words 1
and 3 and the key, and the key is advanced by ``0x9E3779B9`` and ``0xBB67AE85``
before every round but the first.

``tests/test_random_units.py`` checks this reference against ``randomgen``'s
compiled implementation.

The equation functions draw one block per call site and step: counter words
``(site, step mod 2**32, step >> 32, stream)`` and key ``(seed mod 2**32,
seed >> 32)``. ``rand()`` is word 0 as a fraction of ``2**32``; the RTL and kernel
take its top ``fraction`` bits, i.e. ``floor(rand·2**fraction)``. ``randn()`` is the
Box–Muller transform of words 0 and 1, ``sqrt(-2 ln((w0 + 1)/2**32))·cos(2π·w1/2**32)``,
computed in floating point by the Python model and by the integer program
:func:`normal_code` in RTL and kernel (``log``, exact square root, quarter-wave
cosine; ``w1`` truncated to 30 bits of a turn). ``poisson(lam)`` inverts the
Poisson distribution on word 0: the count is the number of thresholds
``ceil(2**32·CDF(k))`` at or below the word (:func:`poisson_thresholds`), so the
Python model, the RTL and the kernel return the same integer, and the sample is
the exact inversion of a 32-bit uniform.
"""

from __future__ import annotations

import bisect
import decimal
import math
from functools import lru_cache

from . import function_units as fu

MULTIPLIER_0 = 0xD2511F53
MULTIPLIER_1 = 0xCD9E8D57
WEYL_0 = 0x9E3779B9
WEYL_1 = 0xBB67AE85
ROUNDS = 10
_MASK = 0xFFFFFFFF

Block = tuple[int, int, int, int]


def philox4x32(counter: Block, key: tuple[int, int], rounds: int = ROUNDS) -> Block:
    """Return the Philox4x32 block for one counter and key.

    Parameters
    ----------
    counter : tuple[int, int, int, int]
        Four 32-bit counter words, least significant first.
    key : tuple[int, int]
        Two 32-bit key words, least significant first.
    rounds : int, optional
        Number of rounds; 10 is the standard Philox4x32-10.

    Returns
    -------
    tuple[int, int, int, int]
        Four 32-bit output words.

    Raises
    ------
    ValueError
        If a word is outside ``[0, 2**32)`` or ``rounds`` is outside ``[1, 16]``.

    """
    if any(not 0 <= word <= _MASK for word in (*counter, *key)):
        raise ValueError("Philox words must be 32-bit unsigned integers")
    if not 1 <= rounds <= 16:
        raise ValueError(f"Philox takes 1 to 16 rounds, got {rounds}")
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(rounds):
        if round_index:
            k0 = (k0 + WEYL_0) & _MASK
            k1 = (k1 + WEYL_1) & _MASK
        product0 = MULTIPLIER_0 * c0
        product1 = MULTIPLIER_1 * c2
        c0, c1, c2, c3 = (
            (product1 >> 32) ^ c1 ^ k0,
            product1 & _MASK,
            (product0 >> 32) ^ c3 ^ k1,
            product0 & _MASK,
        )
    return c0, c1, c2, c3


def counter_words(value: int) -> Block:
    """Split a 128-bit counter into four 32-bit words, least significant first.

    Parameters
    ----------
    value : int
        Counter in ``[0, 2**128)``.

    Returns
    -------
    tuple[int, int, int, int]
        The four words.

    Raises
    ------
    ValueError
        If ``value`` is outside ``[0, 2**128)``.

    """
    if not 0 <= value < 1 << 128:
        raise ValueError("a Philox counter has 128 bits")
    return (value & _MASK, (value >> 32) & _MASK, (value >> 64) & _MASK, value >> 96)


def draw(site: int, step: int, stream: int, seed: int) -> Block:
    """Return the Philox block of one call site at one step.

    Parameters
    ----------
    site : int
        Call-site number within the neuron, ``[0, 2**32)``.
    step : int
        Step number, ``[0, 2**64)``.
    stream : int
        Stream (neuron instance) number, ``[0, 2**32)``.
    seed : int
        Seed, ``[0, 2**64)``.

    Returns
    -------
    tuple[int, int, int, int]
        Four 32-bit words.

    """
    counter = (site & _MASK, step & _MASK, (step >> 32) & _MASK, stream & _MASK)
    return philox4x32(counter, (seed & _MASK, (seed >> 32) & _MASK))


def uniform_value(word: int) -> float:
    """Return ``rand()``: the word as a fraction of ``2**32``, in ``[0, 1)``.

    Parameters
    ----------
    word : int
        A 32-bit output word.

    Returns
    -------
    float
        ``word / 2**32``.

    """
    return word / 4294967296.0


def uniform_code(word: int, fraction: int) -> int:
    """Return the code of ``rand()`` at ``fraction`` bits: ``floor(word / 2**(32 - fraction))``.

    Parameters
    ----------
    word : int
        A 32-bit output word.
    fraction : int
        Fraction bits, ``[0, 32]``.

    Returns
    -------
    int
        Non-negative code below ``2**fraction``.

    """
    return word >> (32 - fraction)


def normal_value(word0: int, word1: int) -> float:
    """Return ``randn()`` in floating point: Box–Muller of two words.

    Parameters
    ----------
    word0, word1 : int
        Two 32-bit output words.

    Returns
    -------
    float
        ``sqrt(-2 ln((word0 + 1)/2**32))·cos(2π·word1/2**32)``.

    """
    radius = math.sqrt(-2.0 * math.log((word0 + 1) / 4294967296.0))
    return radius * math.cos(2.0 * math.pi * word1 / 4294967296.0)


def normal_code(word0: int, word1: int, data_width: int, fraction: int) -> int:
    """Return the code of ``randn()`` computed in integers, as the RTL and kernel compute it.

    Parameters
    ----------
    word0, word1 : int
        Two 32-bit output words.
    data_width, fraction : int
        The output format.

    Returns
    -------
    int
        The result code, saturated to the word.

    """
    log2u = fu.log2_internal(word0 + 1, 32)  # log2((w0 + 1)/2**32), fraction P, <= 0
    whole = log2u >> fu.P
    rest = log2u - (whole << fu.P)
    square = -2 * (whole * fu._LN2 + ((rest * fu._LN2) >> fu.P))  # -2 ln u, fraction P
    radius = fu.isqrt(square << 26)  # fraction 28
    turn = word1 >> 2  # 30 bits of a turn
    cosine = fu._turn_sine((turn + (1 << (fu.B - 2))) & ((1 << fu.B) - 1))  # fraction P
    return fu.saturate(fu.shift_round(radius * cosine, fraction - 58), data_width)


MAX_POISSON_LAMBDA = 100.0


@lru_cache(maxsize=256)
def poisson_thresholds(lam: float) -> tuple[int, ...]:
    """Return the inversion thresholds of a Poisson distribution for 32-bit uniform words.

    Threshold ``k`` is ``ceil(2**32·CDF(k))``; a word ``w`` gives the count
    ``#{k : threshold_k <= w}``, the smallest ``k`` with ``w < 2**32·CDF(k)``. The
    list ends at the first ``k`` whose threshold reaches ``2**32``, which no word
    reaches. The CDF is summed with 50 significant digits (:mod:`decimal`) from the exact
    binary value of ``lam``.

    Parameters
    ----------
    lam : float
        The mean, ``[0, MAX_POISSON_LAMBDA]``.

    Returns
    -------
    tuple[int, ...]
        Non-decreasing thresholds in ``[1, 2**32)``.

    Raises
    ------
    ValueError
        If ``lam`` is not a finite number in ``[0, MAX_POISSON_LAMBDA]``.

    """
    if not (math.isfinite(lam) and 0.0 <= lam <= MAX_POISSON_LAMBDA):
        raise ValueError(f"poisson() mean must be in [0, {MAX_POISSON_LAMBDA:g}], got {lam!r}")
    thresholds: list[int] = []
    context = decimal.Context(prec=50, rounding=decimal.ROUND_HALF_EVEN)
    scale = decimal.Decimal(1 << 32)
    rate = decimal.Decimal(lam)  # the exact binary value of the float
    term = context.exp(-rate)
    cdf = term
    k = 0
    while True:
        threshold = int(context.multiply(cdf, scale).to_integral_value(decimal.ROUND_CEILING))
        if threshold >= 1 << 32:
            return tuple(thresholds)
        thresholds.append(threshold)
        k += 1
        term = context.divide(context.multiply(term, rate), k)
        cdf = context.add(cdf, term)


def poisson_count(word: int, lam: float) -> int:
    """Return the Poisson count of one 32-bit word by inversion.

    Parameters
    ----------
    word : int
        A 32-bit output word.
    lam : float
        The mean, see :func:`poisson_thresholds`.

    Returns
    -------
    int
        The count, ``>= 0``.

    """
    return bisect.bisect_right(poisson_thresholds(lam), word)


def poisson_code(word: int, lam: float, data_width: int, fraction: int) -> int:
    """Return the code of ``poisson(lam)``: the count in the output format, saturated.

    Parameters
    ----------
    word : int
        A 32-bit output word.
    lam : float
        The mean.
    data_width, fraction : int
        The output format.

    Returns
    -------
    int
        ``count·2**fraction``, saturated to the word.

    """
    return fu.saturate(poisson_count(word, lam) << fraction, data_width)


__all__ = [
    "MAX_POISSON_LAMBDA",
    "MULTIPLIER_0",
    "MULTIPLIER_1",
    "ROUNDS",
    "WEYL_0",
    "WEYL_1",
    "counter_words",
    "draw",
    "normal_code",
    "normal_value",
    "philox4x32",
    "poisson_code",
    "poisson_count",
    "poisson_thresholds",
    "uniform_code",
    "uniform_value",
]
