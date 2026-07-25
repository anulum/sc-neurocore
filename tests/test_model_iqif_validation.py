# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IQIF validation and state contracts

"""Constructor, current, mutation, and reset validation tests for IQIF."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models.iqif import IntegerQIFNeuron


@pytest.mark.parametrize(
    "kwargs",
    (
        {"a": -1},
        {"b": -1},
        {"a": 0, "b": 0},
        {"v_min": 128},
        {"v_rest": 200},
        {"v_threshold": 255},
        {"v_reset": 256},
        {"v": -1},
        {"v_max": 1 << 31},
    ),
)
def test_constructor_rejects_invalid_contracts(kwargs: dict[str, int]) -> None:
    """Every ordering, coefficient, state, and width invariant fails closed."""
    with pytest.raises(ValueError):
        IntegerQIFNeuron(**kwargs)


@pytest.mark.parametrize("value", (True, 10.0, 10.5, "10", object()))
def test_current_requires_an_exact_integer_without_mutation(value: object) -> None:
    """No Boolean or Float64 contamination crosses the integer soma boundary."""
    neuron = IntegerQIFNeuron()
    before = neuron.v
    with pytest.raises(ValueError, match="current"):
        neuron.step(cast(int, value))
    assert neuron.v == before


def test_numpy_integer_is_accepted_and_normalised() -> None:
    """Integer scalar protocols remain usable without accepting float coercion."""
    neuron = IntegerQIFNeuron(v=np.int64(128))  # type: ignore[arg-type] # NumPy integer contract
    assert type(neuron.v) is int
    assert neuron.step(np.int32(10)) == 0  # type: ignore[arg-type] # NumPy integer contract
    assert neuron.v == 138


def test_runtime_parameter_corruption_fails_before_state_mutation() -> None:
    """Public dataclass mutation is revalidated on every state-changing call."""
    neuron = IntegerQIFNeuron()
    neuron.a = -1
    before = neuron.v
    with pytest.raises(ValueError, match="non-negative"):
        neuron.step(10)
    assert neuron.v == before


def test_reset_restores_rest_only() -> None:
    """Reset restores v_rest while preserving all configured parameters."""
    neuron = IntegerQIFNeuron(
        v=150,
        v_rest=100,
        v_threshold=180,
        v_reset=140,
        a=2,
        b=7,
        v_max=250,
        v_min=3,
    )
    parameters = (neuron.v_rest, neuron.v_threshold, neuron.v_reset, neuron.a, neuron.b)
    neuron.reset()
    assert neuron.v == 100
    assert (neuron.v_rest, neuron.v_threshold, neuron.v_reset, neuron.a, neuron.b) == parameters
