# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dot-product variance contracts, checked against simulated bitstreams

"""Bind each dot-product variance formula to the circuit it describes.

The reference in these tests is a simulation of the circuit: operand
bitstreams are drawn, combined by AND, and reduced by the adder under test.
The variance of the simulated estimates is compared with the formula. One
case is enumerated exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from sc_neurocore.core import (
    GuaranteeNotDetermined,
    dot_product_variance,
    interleaved_dot_product_variance,
    mux_add_variance,
    mux_dot_product_variance,
)

_REPETITIONS = 20_000
#: Six standard errors of a sample variance over ``_REPETITIONS`` near-normal estimates.
_RELATIVE_TOLERANCE = 6.0 * (2.0 / (_REPETITIONS - 1)) ** 0.5


def _product_streams(
    values_a: list[float], values_b: list[float], length: int, seed: int
) -> np.ndarray:
    """Return simulated AND streams, shape (repetitions, terms, length)."""
    generator = np.random.default_rng(seed)
    shape = (_REPETITIONS, len(values_a), length)
    bits_a = generator.random(shape) < np.asarray(values_a)[None, :, None]
    bits_b = generator.random(shape) < np.asarray(values_b)[None, :, None]
    return bits_a & bits_b


def _averaged(streams: np.ndarray, seed: int) -> np.ndarray:
    """Every product stream is counted in full and the counts are averaged."""
    del seed
    return streams.mean(axis=(1, 2))


def _random_multiplexer(streams: np.ndarray, seed: int) -> np.ndarray:
    """One output stream; each bit is taken from a uniformly drawn term."""
    generator = np.random.default_rng(seed + 1)
    repetitions, terms, length = streams.shape
    select = generator.integers(0, terms, size=(repetitions, length))
    chosen = np.take_along_axis(streams, select[:, None, :], axis=1)[:, 0, :]
    return chosen.mean(axis=1)


def _round_robin(streams: np.ndarray, seed: int) -> np.ndarray:
    """One output stream; bit ``t`` is taken from term ``t mod K``."""
    del seed
    repetitions, terms, length = streams.shape
    select = np.broadcast_to(np.arange(length) % terms, (repetitions, length))
    chosen = np.take_along_axis(streams, select[:, None, :], axis=1)[:, 0, :]
    return chosen.mean(axis=1)


_CIRCUITS: dict[str, tuple[Callable[[np.ndarray, int], np.ndarray], Callable[..., float]]] = {
    "averaged": (_averaged, dot_product_variance),
    "random multiplexer": (_random_multiplexer, mux_dot_product_variance),
    "round robin": (_round_robin, interleaved_dot_product_variance),
}

_VECTORS = {
    "two mixed terms": ([0.9, 0.2], [0.8, 0.7]),
    "four unequal terms": ([0.1, 0.5, 0.9, 0.3], [0.6, 0.4, 0.95, 1.0]),
    "three equal terms": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


@pytest.mark.parametrize("circuit", sorted(_CIRCUITS))
@pytest.mark.parametrize("vectors", sorted(_VECTORS))
def test_formula_matches_the_simulated_circuit(circuit: str, vectors: str) -> None:
    """Each formula predicts the variance of its own circuit's estimates."""
    simulate, formula = _CIRCUITS[circuit]
    values_a, values_b = _VECTORS[vectors]
    length = 12 * len(values_a)
    streams = _product_streams(values_a, values_b, length, seed=20261112)
    estimates = simulate(streams, 20261112)
    expected_mean = float(np.mean(np.asarray(values_a) * np.asarray(values_b)))
    assert float(estimates.mean()) == pytest.approx(expected_mean, abs=5e-3)
    predicted = formula(values_a, values_b, length)
    assert float(estimates.var()) == pytest.approx(predicted, rel=_RELATIVE_TOLERANCE)


def test_formulas_are_not_interchangeable() -> None:
    """The averaged formula does not describe a multiplexer's output."""
    values_a, values_b = _VECTORS["four unequal terms"]
    length = 48
    streams = _product_streams(values_a, values_b, length, seed=7)
    observed = float(_random_multiplexer(streams, 7).var())
    averaged = dot_product_variance(values_a, values_b, length)
    assert observed > 3.0 * averaged


def test_constant_zero_and_one_through_a_random_multiplexer() -> None:
    """Published counterexample: terms 0 and 1 give 1/(4N), not zero."""
    assert mux_dot_product_variance([0.0, 1.0], [1.0, 1.0], 100) == pytest.approx(0.0025)
    assert dot_product_variance([0.0, 1.0], [1.0, 1.0], 100) == 0.0
    assert interleaved_dot_product_variance([0.0, 1.0], [1.0, 1.0], 100) == 0.0


def test_random_multiplexer_of_two_constants_enumerated_exactly() -> None:
    """For two bits every selector outcome is listed and the variance is 1/8."""
    outcomes = [Fraction(sum(selection), 2) for selection in product((0, 1), repeat=2)]
    mean = sum(outcomes, Fraction(0)) / len(outcomes)
    variance = sum(((value - mean) ** 2 for value in outcomes), Fraction(0)) / len(outcomes)
    assert variance == Fraction(1, 8)
    assert mux_dot_product_variance([0.0, 1.0], [1.0, 1.0], 2) == pytest.approx(float(variance))


def test_two_term_multiplexer_agrees_with_the_scaled_adder() -> None:
    """With unit weights the two-term multiplexer is the fair-select adder."""
    assert mux_dot_product_variance([0.3, 0.7], [1.0, 1.0], 128) == pytest.approx(
        mux_add_variance(0.3, 0.7, 128)
    )


@pytest.mark.parametrize("vectors", sorted(_VECTORS))
def test_order_of_the_three_variances(vectors: str) -> None:
    """Averaging is never worse than round robin, which is never worse than random."""
    values_a, values_b = _VECTORS[vectors]
    length = 12 * len(values_a)
    averaged = dot_product_variance(values_a, values_b, length)
    round_robin = interleaved_dot_product_variance(values_a, values_b, length)
    random = mux_dot_product_variance(values_a, values_b, length)
    assert averaged <= round_robin <= random


def test_single_term_makes_all_three_equal() -> None:
    """With one term there is nothing to select and the three contracts coincide."""
    expected = 0.42 * 0.58 / 50
    for formula in (
        dot_product_variance,
        mux_dot_product_variance,
        interleaved_dot_product_variance,
    ):
        assert formula([0.7], [0.6], 50) == pytest.approx(expected)


def test_round_robin_needs_a_whole_number_of_rounds() -> None:
    """A length that is not a multiple of the term count is refused."""
    with pytest.raises(GuaranteeNotDetermined, match="multiple"):
        interleaved_dot_product_variance([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 10)


@pytest.mark.parametrize("formula", [mux_dot_product_variance, interleaved_dot_product_variance])
def test_invalid_vectors_are_refused(formula: Callable[..., float]) -> None:
    """Unequal lengths, empty vectors and values outside [0, 1] are refused."""
    with pytest.raises(ValueError):
        formula([0.5, 0.5], [0.5], 10)
    with pytest.raises(ValueError):
        formula([], [], 10)
    with pytest.raises(ValueError):
        formula([1.5], [0.5], 10)
    with pytest.raises(ValueError):
        formula([0.5], [0.5], 0)
