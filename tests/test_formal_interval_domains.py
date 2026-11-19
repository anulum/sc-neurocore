# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The interval checker holds only on inputs inside its domains

"""Hold the interval checker's verdicts to inputs inside their domains.

The probability check once said ``HELD`` for negative "probabilities", because a product of two negative intervals
lies in ``[0, 1]``, and the energy check said ``HELD`` for a negative energy above a more negative cost. An interval
could also be built with its ends reversed or with NaN. These tests hold well-formed intervals, a probability verdict
only for intervals inside ``[0, 1]``, and an energy verdict only for finite, non-negative energy and cost.
"""

from __future__ import annotations

import logging
import math

import pytest

from sc_neurocore.verification.formal_proofs import FormalVerifier, Interval


@pytest.mark.parametrize(
    ("low", "high"),
    [(1.0, 0.0), (math.nan, 1.0), (0.0, math.nan), (-math.inf, 0.0), (0.0, math.inf)],
)
def test_an_interval_needs_finite_ends_in_order(low: float, high: float) -> None:
    """Reversed ends, NaN and infinity make no interval."""
    with pytest.raises(ValueError, match="interval"):
        Interval(low, high)


def test_general_intervals_keep_negative_and_single_point_ranges() -> None:
    """Interval arithmetic itself is not limited to probabilities."""
    assert Interval(-2.0, 3.0) * Interval(4.0, 5.0) == Interval(-10.0, 15.0)
    assert Interval(0.5, 0.5) + Interval(-0.5, -0.5) == Interval(0.0, 0.0)


def test_an_arithmetic_overflow_is_refused_rather_than_carried_as_infinity() -> None:
    """A sum beyond the float range has no finite end, so it is refused."""
    with pytest.raises(ValueError, match="interval"):
        Interval(1e308, 1e308) + Interval(1e308, 1e308)


@pytest.mark.parametrize(
    ("p_in", "p_weight"),
    [
        (Interval(-0.5, -0.2), Interval(-0.5, -0.2)),
        (Interval(-0.1, 0.5), Interval(0.0, 1.0)),
        (Interval(0.0, 1.0), Interval(0.2, 1.5)),
    ],
)
def test_the_probability_bound_is_not_shown_when_an_input_is_not_a_probability(
    p_in: Interval, p_weight: Interval, caplog: pytest.LogCaptureFixture
) -> None:
    """Both operands must lie in ``[0, 1]``; otherwise the bound is not shown, whatever the product is."""
    with caplog.at_level(logging.INFO, logger="sc_neurocore.verification.formal_proofs"):
        assert FormalVerifier.verify_probability_bounds(p_in, p_weight) is False
    assert "NOT SHOWN" in caplog.text and "not a probability interval" in caplog.text


def test_the_probability_verdict_holds_for_probability_intervals() -> None:
    """Two intervals inside ``[0, 1]`` multiply into ``[0, 1]``, including the closed ends."""
    assert FormalVerifier.verify_probability_bounds(Interval(0.0, 1.0), Interval(0.0, 1.0)) is True
    assert FormalVerifier.verify_probability_bounds(Interval(0.1, 0.8), Interval(0.3, 0.3)) is True


@pytest.mark.parametrize(
    ("energy", "cost"),
    [(-5.0, -10.0), (10.0, -1.0), (-1.0, 0.0), (math.nan, 1.0), (math.inf, 1.0), (5.0, math.inf)],
)
def test_the_energy_bound_is_not_shown_outside_finite_non_negative_values(
    energy: float, cost: float, caplog: pytest.LogCaptureFixture
) -> None:
    """A negative, NaN or infinite energy or cost shows nothing about the energy left."""
    with caplog.at_level(logging.WARNING, logger="sc_neurocore.verification.formal_proofs"):
        assert FormalVerifier.verify_energy_safety(energy, cost) is False
    assert "NOT SHOWN" in caplog.text and "finite and non-negative" in caplog.text


def test_the_energy_verdict_holds_with_enough_energy_and_fails_on_overdraw() -> None:
    """Zero cost and an exact budget hold; a cost above the energy does not."""
    assert FormalVerifier.verify_energy_safety(0.0, 0.0) is True
    assert FormalVerifier.verify_energy_safety(3.0, 3.0) is True
    assert FormalVerifier.verify_energy_safety(2.0, 3.0) is False
