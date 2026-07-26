# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka source dynamics tests

"""Branch, event, protocol-count, and operating-envelope contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron
from tests.model_ibarz_tanaka_support import _reference_step


@pytest.mark.parametrize("v", (-2.0, -1.0, 0.5, 1.5))
def test_all_four_fast_branches_match_eq_3(v: float) -> None:
    """Each source branch commits the independently evaluated candidate."""
    neuron = IbarzTanakaMapNeuron(v=v)
    expected = _reference_step(neuron, 0.2)
    assert neuron.step(0.2) == expected[2]
    assert (neuron.v, neuron.u) == expected[:2]


def test_slow_state_uses_the_pre_step_fast_state() -> None:
    """The Eq. 2 update is simultaneous with the Eq. 3 fast update."""
    neuron = IbarzTanakaMapNeuron(v=0.5, u=-0.1)
    expected_u = neuron.u - neuron.mu * (neuron.v + 1.0 - neuron.sigma)
    neuron.step(0.2)
    assert neuron.u == expected_u


def test_event_marks_the_source_reset_branch() -> None:
    """The plateau branch precedes a reset event on the next iteration."""
    neuron = IbarzTanakaMapNeuron(v=0.5, u=-0.1)
    assert neuron.step(0.2) == 0
    assert neuron.v == pytest.approx(1.1)
    assert neuron.step(0.2) == 1
    assert neuron.v == -1.0


@pytest.mark.parametrize("current, expected", ((0.0, 9), (0.2, 33), (1.0, 195)))
def test_source_protocol_event_counts(current: float, expected: int) -> None:
    """The published default parameter set has stable derived event counts."""
    _trace, events = IbarzTanakaMapNeuron().simulate(1000, current, backend="python")
    assert events == expected


@pytest.mark.parametrize("current", (-1.0, 0.0, 0.2, 1.0, 10.0))
def test_long_run_remains_finite(current: float) -> None:
    """The source operating envelope keeps both state variables finite."""
    neuron = IbarzTanakaMapNeuron()
    trace, _events = neuron.simulate(10_000, current, backend="python")
    assert np.isfinite(trace).all()
    assert math.isfinite(neuron.u)
