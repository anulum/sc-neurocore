# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained upward-crossing Rulkov map contracts

"""Public contracts for the retained SC Rulkov event identity."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)


def test_retained_identity_preserves_state_but_not_source_event_timing() -> None:
    """The two identities share recurrence while exposing distinct events."""
    source = RulkovMapNeuron()
    retained = SCUpwardCrossingRulkovMapNeuron()

    assert [source.step(2.0) for _ in range(3)] == [0, 0, 1]
    assert [retained.step(2.0) for _ in range(3)] == [1, 0, 0]
    assert (retained.x, retained.y) == (source.x, source.y)


def test_configurable_threshold_changes_only_the_observation_surface() -> None:
    """A higher SC threshold delays the event without changing map state."""
    zero = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.0)
    high = SCUpwardCrossingRulkovMapNeuron(x_threshold=2.0)

    zero_events = [zero.step(2.0) for _ in range(3)]
    high_events = [high.step(2.0) for _ in range(3)]

    assert zero_events == [1, 0, 0]
    assert high_events == [0, 1, 0]
    assert (zero.x, zero.y) == (high.x, high.y)


def test_public_batch_matches_public_step_sequence() -> None:
    """Batch execution must be the exact public-step recurrence closure."""
    stepped = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
    expected_trace = np.empty(512, dtype=np.float64)
    expected_events = 0
    for index in range(expected_trace.size):
        expected_events += stepped.step(0.5)
        expected_trace[index] = stepped.x

    batched = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
    trace, events = batched.simulate(512, 0.5, backend="python")

    np.testing.assert_array_equal(trace, expected_trace)
    assert events == expected_events
    assert (batched.x, batched.y) == (stepped.x, stepped.y)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_non_finite_threshold_is_rejected(threshold: float) -> None:
    """The observation threshold must be a finite scalar."""
    with pytest.raises(ValueError, match="x_threshold must be finite"):
        SCUpwardCrossingRulkovMapNeuron(x_threshold=threshold)


def test_invalid_runtime_state_is_failure_atomic() -> None:
    """A bad state must fail before either coordinate is mutated."""
    neuron = SCUpwardCrossingRulkovMapNeuron()
    neuron.y = np.inf
    before = (neuron.x, neuron.y)

    with pytest.raises(FloatingPointError, match="state must be finite"):
        neuron.simulate(4, 0.5, backend="python")

    assert (neuron.x, neuron.y) == before
