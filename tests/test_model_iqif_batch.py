# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF batch simulation contracts

"""Batch dispatch, validation, and firing-rate tests for IQIF."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron


def test_python_batch_is_exact_and_commits_final_state() -> None:
    """The public Python batch returns contiguous int64 state and exact events."""
    neuron = IntegerQIFNeuron()
    trace, spikes = neuron.simulate(400, 10, backend="python")
    assert trace.dtype == np.int64
    assert trace.flags.c_contiguous
    assert trace.shape == (400,)
    assert spikes == 26
    assert neuron.v == trace[-1] == 198


def test_empty_batch_preserves_state() -> None:
    """Zero ticks allocate no phantom state or event."""
    neuron = IntegerQIFNeuron(v=150)
    trace, spikes = neuron.simulate(0, 10, backend="python")
    assert trace.shape == (0,)
    assert spikes == 0
    assert neuron.v == 150


@pytest.mark.parametrize("n_steps", (-1, True, 1.0, 1 << 31))
def test_invalid_batch_length_fails_before_mutation(n_steps: object) -> None:
    """The public batch has one bounded integer length contract."""
    neuron = IntegerQIFNeuron()
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 10)
    assert neuron.v == 128


def test_invalid_backend_fails_before_mutation() -> None:
    """Unknown dispatch selectors cannot silently fall through to Python."""
    neuron = IntegerQIFNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 10, backend="cuda")
    assert neuron.v == 128


def test_firing_rate_is_monotonic_over_the_source_tutorial_currents() -> None:
    """The enrolled tonic regime increases event count with integer drive."""
    counts = []
    for current in (5, 10, 20, 40):
        _, spikes = IntegerQIFNeuron().simulate(3_000, current, backend="python")
        counts.append(spikes)
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
