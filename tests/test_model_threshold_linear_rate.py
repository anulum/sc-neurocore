# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end threshold-linear rate model tests

"""Validate the memoryless ``gain * max(0, I-theta)`` rate contract."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


def test_defaults_and_float_output() -> None:
    neuron = ThresholdLinearRateNeuron()
    assert (neuron.r, neuron.theta, neuron.gain) == (0.0, 0.0, 1.0)
    assert isinstance(neuron.step(1.0), float)


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1.0, 0.0), (1.5, 0.0), (3.0, 3.0), (-4.0, 0.0)],
)
def test_piecewise_linear_branches(current: float, expected: float) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    assert neuron.step(current) == expected
    assert neuron.r == expected


def test_output_is_memoryless() -> None:
    neuron = ThresholdLinearRateNeuron(theta=1.5, gain=2.0)
    assert neuron.step(3.0) == 3.0
    assert neuron.step(1.0) == 0.0


def test_reset_preserves_configuration() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=-0.4, gain=2.5)
    neuron.step(3.0)
    neuron.reset()
    assert (neuron.r, neuron.theta, neuron.gain) == (0.0, -0.4, 2.5)


def test_python_batch_matches_scalar_steps() -> None:
    scalar = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    expected = np.asarray([scalar.step(3.0) for _ in range(32)])
    batched = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    actual = batched.simulate(32, 3.0, backend="python")
    np.testing.assert_array_equal(actual, expected)
    assert batched.r == scalar.r


def test_empty_batch_preserves_cached_output() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    np.testing.assert_array_equal(neuron.simulate(0, 3.0, backend="python"), np.empty(0))
    assert neuron.r == 0.25


def test_descriptor_tracks_parameters_and_algebraic_scope() -> None:
    payload = load_descriptor_payload("ThresholdLinearRateNeuron")
    assert payload is not None
    assert set(payload["state"]) == {"r"}
    assert set(payload["parameters"]) == {"theta", "gain"}
    assert payload["integration"] == {"dt": 1.0, "method": "map"}
    assert set(payload["backends"]) == {"python", "rust", "julia", "go", "mojo"}
    assert "no ODE" in payload["dynamics"]["scope"]


def test_schema_map_matches_hand_model() -> None:
    configured = {"theta": 1.5, "gain": 2.0}
    schema = UniversalNeuron.from_schema("threshold_linear_rate", parameter_overrides=configured)
    hand = ThresholdLinearRateNeuron(**configured)
    currents = [1.0, 1.5, 3.0, -4.0]
    schema_trace: list[float] = []
    hand_trace: list[float] = []
    for current in currents:
        schema.step(I=current)
        schema_trace.append(schema.state["r"])
        hand_trace.append(hand.step(current))
    np.testing.assert_array_equal(schema_trace, hand_trace)


@pytest.mark.parametrize("field", ["r", "theta", "gain"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_constructor_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ThresholdLinearRateNeuron(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("r", -1.0), ("gain", -1.0)])
def test_rejects_negative_rate_or_gain(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ThresholdLinearRateNeuron(**{field: value})


@pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_current_without_mutation(current: float) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert neuron.r == 0.25


@pytest.mark.parametrize("field", ["r", "theta", "gain"])
def test_rejects_corrupted_runtime_contract_without_mutation(field: str) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, theta=1.5, gain=2.0)
    setattr(neuron, field, np.nan)
    with pytest.raises(ValueError, match=field):
        neuron.step(3.0)
    if field != "r":
        assert neuron.r == 0.25


def test_rejects_overflowing_output_without_mutation() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25, gain=1.0e308)
    with pytest.raises(ValueError, match="rate output"):
        neuron.step(1.0e308)
    assert neuron.r == 0.25


@pytest.mark.parametrize("n_steps", [-1, 1.5, True])
def test_rejects_invalid_batch_length_without_mutation(n_steps: object) -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 3.0, backend="python")
    assert neuron.r == 0.25


def test_rejects_unknown_backend_without_mutation() -> None:
    neuron = ThresholdLinearRateNeuron(r=0.25)
    with pytest.raises(ValueError, match="backend must be"):
        neuron.simulate(1, 3.0, backend="cuda")
    assert neuron.r == 0.25


def test_deterministic_and_population_compatible() -> None:
    first = ThresholdLinearRateNeuron(theta=1.0, gain=2.0).simulate(100, 3.0, backend="python")
    second = ThresholdLinearRateNeuron(theta=1.0, gain=2.0).simulate(100, 3.0, backend="python")
    np.testing.assert_array_equal(first, second)
    assert Population(ThresholdLinearRateNeuron, n=10, label="tlr").n == 10
