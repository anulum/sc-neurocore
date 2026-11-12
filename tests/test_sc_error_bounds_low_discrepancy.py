# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deterministic error bounds of threshold encoding, checked by enumeration

"""Bind the deterministic bounds of threshold encoding to the points actually used.

The reference in these tests is an enumeration: every bound is compared with the
largest error found by encoding many values with the same points. Nothing here
restates the formula under test.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import qmc

from sc_neurocore.core import (
    GuaranteeNotDetermined,
    low_discrepancy_error_bound,
    threshold_encoding_error_bound,
)


def _radical_inverse_base_2(index: int) -> float:
    """Return the base-2 radical inverse of ``index`` by reversing its bits."""
    value, weight = 0.0, 0.5
    while index:
        if index & 1:
            value += weight
        index >>= 1
        weight *= 0.5
    return value


def _largest_observed_error(points: list[float]) -> float:
    """Encode many values with ``points`` and return the largest decoding error.

    The values are a fine grid, every point itself, and the next representable
    number above every point, where the error of a threshold code jumps.
    """
    count = len(points)
    candidates = [step / 4096 for step in range(4097)]
    candidates += points
    candidates += [math.nextafter(point, 2.0) for point in points]
    worst = 0.0
    for value in candidates:
        if not 0.0 <= value <= 1.0:
            continue
        ones = sum(point < value for point in points)
        worst = max(worst, abs(ones / count - value))
    return worst


_HALTON_BASE_2_PREFIX = [0.0, 0.5, 0.25, 0.75, 0.125]


def test_published_counterexample_exceeds_one_over_n() -> None:
    """Five Halton points encode 0.3 as 0.6: the error 0.3 is above 1/5."""
    ones = sum(point < 0.3 for point in _HALTON_BASE_2_PREFIX)
    assert ones / 5 - 0.3 == pytest.approx(0.3)
    assert 0.3 > 1 / 5


def test_length_five_has_no_one_over_n_guarantee() -> None:
    """A length that is not a power of two is refused, not answered with 1/N."""
    with pytest.raises(GuaranteeNotDetermined, match="power of two"):
        low_discrepancy_error_bound(5)


@pytest.mark.parametrize("length", [3, 5, 6, 7, 9, 100, 255, 257, 1000])
def test_lengths_that_are_not_powers_of_two_are_refused(length: int) -> None:
    """The closed form is stated only where its assumption holds."""
    with pytest.raises(GuaranteeNotDetermined):
        low_discrepancy_error_bound(length)


@pytest.mark.parametrize("exponent", range(0, 11))
def test_power_of_two_prefix_keeps_one_over_n(exponent: int) -> None:
    """For 2**k radical-inverse points the enumerated error stays below 1/N."""
    length = 2**exponent
    points = [_radical_inverse_base_2(index) for index in range(length)]
    bound = low_discrepancy_error_bound(length)
    assert bound == 1.0 / length
    assert _largest_observed_error(points) <= bound
    assert threshold_encoding_error_bound(points) == pytest.approx(bound, abs=1e-15)


@pytest.mark.parametrize("length", range(1, 65))
def test_exact_bound_equals_the_enumerated_worst_case(length: int) -> None:
    """For every prefix length the bound is reached and never exceeded."""
    points = [_radical_inverse_base_2(index) for index in range(length)]
    bound = threshold_encoding_error_bound(points)
    observed = _largest_observed_error(points)
    assert observed <= bound + 1e-12
    assert bound - observed <= 1e-9


def test_exact_bound_of_the_published_counterexample() -> None:
    """The five Halton points have the exact bound 0.35, which covers 0.3."""
    assert threshold_encoding_error_bound(_HALTON_BASE_2_PREFIX) == pytest.approx(0.35)


@pytest.mark.parametrize("length", [5, 12, 37, 64, 100])
def test_reference_sobol_and_halton_prefixes_obey_their_exact_bound(length: int) -> None:
    """Unscrambled library sequences are bounded by the bound of their own prefix."""
    with warnings.catch_warnings():
        # SciPy warns that a prefix which is not a power of two loses balance;
        # that loss is what this test measures.
        warnings.simplefilter("ignore", UserWarning)
        sobol = qmc.Sobol(d=1, scramble=False).random(length)[:, 0].tolist()
    halton_base_3 = qmc.Halton(d=2, scramble=False).random(length)[:, 1].tolist()
    for points in (sobol, halton_base_3):
        bound = threshold_encoding_error_bound(points)
        assert _largest_observed_error(points) <= bound + 1e-12
        assert bound >= 1.0 / (2 * length) - 1e-15


def test_random_points_obey_their_exact_bound() -> None:
    """The bound holds for points with no structure at all."""
    generator = np.random.default_rng(20261112)
    for length in (1, 2, 17, 200):
        points = generator.random(length).tolist()
        bound = threshold_encoding_error_bound(points)
        observed = _largest_observed_error(points)
        assert observed <= bound + 1e-12
        assert bound - observed <= 1e-9


def test_centred_grid_reaches_the_smallest_possible_bound() -> None:
    """The points (2i - 1) / (2N) give 1/(2N), and no set does better."""
    length = 8
    points = [(2 * index + 1) / (2 * length) for index in range(length)]
    assert threshold_encoding_error_bound(points) == pytest.approx(1 / (2 * length))


def test_point_order_does_not_change_the_bound() -> None:
    """The bound depends on the set of points, not on their order."""
    points = [0.75, 0.0, 0.125, 0.5, 0.25]
    assert threshold_encoding_error_bound(points) == threshold_encoding_error_bound(sorted(points))


@pytest.mark.parametrize(
    "points",
    [[], [0.5, math.nan], [0.5, math.inf], [-0.1, 0.5], [0.5, 1.0], [0.5, 1.5]],
)
def test_point_sets_outside_the_unit_interval_are_refused(points: list[float]) -> None:
    """An empty set, a non-finite point or a point outside [0, 1) is refused."""
    with pytest.raises(ValueError):
        threshold_encoding_error_bound(points)


def test_the_refusal_is_a_value_error_with_its_own_type() -> None:
    """A caller can catch the missing guarantee by its type or as a value error."""
    assert issubclass(GuaranteeNotDetermined, ValueError)
