# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Temporal property checks decide only what the trace shows

"""Hold the temporal property checks to verdicts the bounded trace supports.

A window longer than the trace once gave ``VERIFIED`` for any trace, even one firing at every step. A latency of
zero rejected an immediate response, a response exactly at the deadline did not count, and negative neuron ids and
stimulus times wrapped around to the end of the trace. These tests hold the inclusive deadline, ``UNKNOWN`` where
the trace ends before a question is settled, a ``VIOLATED`` that a short trace already proves, and the refusal of
ids, times and sizes outside the trace.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from sc_neurocore.verification.temporal_properties import (
    PropertyResult,
    bounded_activity,
    causal_order,
    fires_within,
    mutual_exclusion,
    rate_bound,
    refractory_guarantee,
)

VERIFIED, VIOLATED, UNKNOWN = (
    PropertyResult.VERIFIED,
    PropertyResult.VIOLATED,
    PropertyResult.UNKNOWN,
)


def _trace(steps: int = 20, neurons: int = 3) -> np.ndarray[Any, Any]:
    """Return a silent spike raster of ``steps`` x ``neurons``."""
    return np.zeros((steps, neurons), dtype=np.int8)


def test_a_response_at_the_stimulus_step_meets_a_zero_latency() -> None:
    """Latency 0 asks for a spike at the stimulus step itself."""
    s = _trace()
    s[4, 0] = 1
    assert fires_within(s, 0, [4], max_latency=0).result is VERIFIED
    assert fires_within(s, 0, [3], max_latency=0).result is VIOLATED


def test_a_response_exactly_at_the_deadline_counts() -> None:
    """The window is ``[t, t + max_latency]``, both ends included."""
    s = _trace()
    s[15, 0] = 1
    assert fires_within(s, 0, [10], max_latency=5).result is VERIFIED
    assert fires_within(s, 0, [10], max_latency=4).result is VIOLATED


def test_a_deadline_past_the_trace_end_without_a_response_is_unknown() -> None:
    """The trace cannot show a response after its last step, so the check does not decide."""
    s = _trace(steps=12)
    result = fires_within(s, 0, [10], max_latency=5)
    assert result.result is UNKNOWN
    assert result.counterexample is None
    assert "after the trace ends" in result.message


def test_a_proven_miss_outweighs_an_undecided_stimulus() -> None:
    """One stimulus whose whole window shows no response makes the property violated."""
    s = _trace(steps=12)
    result = fires_within(s, 0, [2, 10], max_latency=3)
    assert result.result is VIOLATED
    assert result.counterexample is not None and result.counterexample.timestep == 2


@pytest.mark.parametrize("stimulus", [-1, 20])
def test_a_stimulus_outside_the_trace_is_refused(stimulus: int) -> None:
    """A negative time would wrap to the trace end; a time past it lies outside the trace."""
    with pytest.raises(ValueError, match="stimulus time"):
        fires_within(_trace(), 0, [stimulus], max_latency=2)


def test_a_negative_latency_is_refused() -> None:
    """No response can come before its stimulus."""
    with pytest.raises(ValueError, match="max_latency"):
        fires_within(_trace(), 0, [1], max_latency=-1)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: fires_within(s, -1, [0], 1),
        lambda s: mutual_exclusion(s, [0, 3]),
        lambda s: rate_bound(s, 3, 0.5, 2),
        lambda s: refractory_guarantee(s, -1, 2),
        lambda s: causal_order(s, 0, -2, 1),
        lambda s: bounded_activity(s, [-1, 1], 2, 1),
    ],
)
def test_a_neuron_id_outside_the_raster_is_refused(
    call: Callable[[np.ndarray[Any, Any]], object],
) -> None:
    """Negative ids would wrap to the last neurons; ids past the raster do not exist."""
    with pytest.raises(ValueError, match="neuron id"):
        call(_trace())


def test_a_raster_that_is_not_two_dimensional_is_refused() -> None:
    """The checks read a ``(T, N)`` raster."""
    with pytest.raises(ValueError, match="shape"):
        rate_bound(np.zeros(10), 0, 0.5, 2)


@pytest.mark.parametrize("window", [0, -3])
def test_a_window_below_one_step_is_refused(window: int) -> None:
    """A window must hold at least one step; zero would divide by zero."""
    with pytest.raises(ValueError, match="window_size"):
        rate_bound(_trace(), 0, 0.5, window)
    with pytest.raises(ValueError, match="window_size"):
        bounded_activity(_trace(), [0], window, 1)


def test_a_window_longer_than_the_trace_is_never_verified() -> None:
    """With no full window, a quiet trace leaves the bound undecided, and a busy one already breaks it."""
    quiet = _trace(steps=5)
    busy = np.ones((5, 3), dtype=np.int8)
    assert rate_bound(quiet, 0, max_rate=0.0, window_size=10).result is UNKNOWN
    assert bounded_activity(quiet, [0, 1], window_size=10, max_total_spikes=0).result is UNKNOWN
    violated = rate_bound(busy, 0, max_rate=0.0, window_size=10)
    assert violated.result is VIOLATED
    assert violated.counterexample is not None and violated.counterexample.timestep == 0
    assert bounded_activity(busy, [0, 1], window_size=10, max_total_spikes=9).result is VIOLATED


def test_a_negative_causal_delay_is_refused() -> None:
    """A cause cannot be looked for after its effect."""
    with pytest.raises(ValueError, match="max_delay"):
        causal_order(_trace(), 0, 1, max_delay=-1)


def test_a_cause_window_reaching_before_the_trace_start_is_unknown() -> None:
    """A B spike earlier than ``max_delay`` with no A in the recorded part may have a cause before the recording."""
    s = _trace()
    s[2, 1] = 1
    early = causal_order(s, 0, 1, max_delay=5)
    assert early.result is UNKNOWN
    assert early.counterexample is None
    assert "before the trace does" in early.message
    s[1, 0] = 1
    assert causal_order(s, 0, 1, max_delay=5).result is VERIFIED


def test_a_full_in_trace_miss_outweighs_an_undecided_early_spike() -> None:
    """A B spike whose whole window lies in the trace without an A spike makes the property violated."""
    s = _trace()
    s[2, 1] = 1
    s[12, 1] = 1
    result = causal_order(s, 0, 1, max_delay=5)
    assert result.result is VIOLATED
    assert result.counterexample is not None and result.counterexample.timestep == 12
