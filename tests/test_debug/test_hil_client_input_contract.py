# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the HIL client refuses thresholds and refills that would silently misbehave

"""Hold the HIL client's input contract for trigger thresholds and token refills."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from sc_neurocore.debug.hil_client import RateLimiter, SpikeEvent, TriggerCondition

_THRESHOLDS: list[tuple[str, Callable[[float], TriggerCondition]]] = [
    ("min_correlation", lambda value: TriggerCondition(min_correlation=value)),
    ("max_precision", lambda value: TriggerCondition(max_precision=value)),
]


@pytest.mark.parametrize("value", [-0.5, -1e-12, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(("field", "make"), _THRESHOLDS, ids=[name for name, _ in _THRESHOLDS])
def test_a_threshold_that_would_silently_disable_its_test_is_refused(
    field: str, make: Callable[[float], TriggerCondition], value: float
) -> None:
    """A negative or non-finite threshold is refused at construction, naming the threshold.

    Parameters
    ----------
    field : str
        The threshold set.
    make : callable
        Builds a trigger with that threshold at the given value.
    value : float
        A negative or non-finite value, which used to disable the test without a word.
    """
    with pytest.raises(ValueError, match=field):
        make(value)


def test_zero_still_disables_a_test_and_positive_thresholds_keep_their_meaning() -> None:
    """0 is the one value that disables a test; a positive threshold fires at its bound."""
    assert not TriggerCondition().evaluate(SpikeEvent(correlation=1.0, precision=0.0))
    assert TriggerCondition(min_correlation=0.5).evaluate(SpikeEvent(correlation=0.5))
    assert TriggerCondition(max_precision=0.25).evaluate(SpikeEvent(precision=0.25))
    assert not TriggerCondition(max_precision=0.25).evaluate(SpikeEvent(precision=0.3))


def test_a_negative_refill_is_refused_and_takes_no_token() -> None:
    """A refill never takes tokens away; a refused one leaves the bucket as it was."""
    limiter = RateLimiter(3)
    assert limiter.allow()
    with pytest.raises(ValueError, match="not negative"):
        limiter.refill(-5)
    assert limiter.available == 2


def test_a_refill_of_zero_changes_nothing_and_a_refill_stops_at_the_capacity() -> None:
    """Zero adds nothing; a large refill fills the bucket to its capacity and no further."""
    limiter = RateLimiter(2)
    assert limiter.allow()
    limiter.refill(0)
    assert limiter.available == 1
    limiter.refill(10)
    assert limiter.available == 2
