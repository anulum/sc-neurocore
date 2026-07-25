# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque public validation contracts

from __future__ import annotations

import math
from typing import cast

import pytest

from sc_neurocore.neurons.models.lapicque import LapicqueNeuron


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Reject negative and non-integer step counts at the public boundary."""
    neuron = LapicqueNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert neuron.v == before


def test_invalid_backend_fails_before_mutation() -> None:
    """Reject unknown dispatch selectors instead of silently using Python."""
    neuron = LapicqueNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    assert neuron.v == 0.0


def test_non_finite_current_fails_before_mutation() -> None:
    """Apply the finite-input boundary to every dispatcher path."""
    neuron = LapicqueNeuron(v=0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan, backend="auto")
    assert neuron.v == 0.25


def test_threshold_must_exceed_reset_before_dispatch() -> None:
    """Reject reset geometry independently of the resting-potential guard."""
    with pytest.raises(ValueError, match="v_threshold must be greater than v_reset"):
        LapicqueNeuron(v_rest=0.0, v_reset=1.0, v_threshold=1.0)
