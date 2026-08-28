# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Teeter GLIF5 behavioral contracts

"""Independent equation and public-surface tests for canonical GLIF5."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.glif import GLIFNeuron


def _state(neuron: GLIFNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.theta_spike,
        neuron.i_asc1,
        neuron.i_asc2,
        neuron.theta_voltage,
        neuron.refractory_remaining,
    )


def _source_interval(neuron: GLIFNeuron, current: float) -> tuple[float, ...]:
    """Evaluate the Allen exact-dynamics update without production helpers."""
    membrane_rate = 1.0 / (neuron.resistance * neuron.capacitance)
    membrane_decay = math.exp(-membrane_rate * neuron.dt)
    voltage_decay = math.exp(-neuron.b_voltage * neuron.dt)
    equilibrium = neuron.resistance * (current + neuron.i_asc1 + neuron.i_asc2)
    offset = neuron.v - neuron.e_l
    next_v = neuron.e_l + equilibrium + (offset - equilibrium) * membrane_decay
    difference = neuron.b_voltage - membrane_rate
    if abs(difference) <= 1e-12 * max(1.0, neuron.b_voltage, membrane_rate):
        convolution = neuron.dt * math.exp(-neuron.b_voltage * neuron.dt)
    else:
        convolution = (math.exp(-membrane_rate * neuron.dt) - voltage_decay) / difference
    forcing = (
        equilibrium * (1.0 - voltage_decay) / neuron.b_voltage
        + (offset - equilibrium) * convolution
    )
    return (
        next_v,
        neuron.theta_spike * math.exp(-neuron.b_spike * neuron.dt),
        neuron.i_asc1 * math.exp(-neuron.k_asc1 * neuron.dt),
        neuron.i_asc2 * math.exp(-neuron.k_asc2 * neuron.dt),
        neuron.theta_voltage * voltage_decay + neuron.a_voltage * forcing,
        0.0,
    )


def test_five_source_states_match_independent_exact_interval() -> None:
    neuron = GLIFNeuron(
        v=-68.0,
        theta_spike=1.25,
        i_asc1=0.4,
        i_asc2=-0.2,
        theta_voltage=0.75,
    )
    expected = _source_interval(neuron, 4.0)

    assert neuron.step(4.0) == 0

    np.testing.assert_allclose(_state(neuron), expected, rtol=0.0, atol=2e-15)


def test_event_condition_is_strict_at_equal_threshold() -> None:
    neuron = GLIFNeuron(v=-50.0, theta_inf=-50.0, a_voltage=0.0)

    assert neuron.step(20.0) == 0
    assert neuron.v == -50.0
    assert neuron.theta == -50.0


def test_source_affine_reset_and_refractory_cut_are_explicit() -> None:
    neuron = GLIFNeuron(
        v=-51.0,
        f_v=0.25,
        delta_v=1.5,
        delta_theta_spike=3.0,
        f_asc1=0.5,
        f_asc2=-0.5,
        delta_i_asc1=1.25,
        delta_i_asc2=-0.25,
        refractory_period=2.0,
    )
    candidate = _source_interval(neuron, 50.0)

    assert neuron.step(50.0) == 1
    assert neuron.v == neuron.e_l + 0.25 * (candidate[0] - neuron.e_l) - 1.5
    assert neuron.theta_spike == candidate[1] + 3.0
    assert neuron.i_asc1 == 0.5 * candidate[2] + 1.25
    assert neuron.i_asc2 == -0.5 * candidate[3] - 0.25
    assert neuron.theta_voltage == candidate[4]
    post_cut = _state(neuron)
    assert neuron.step(1e6) == 0
    assert _state(neuron)[:-1] == post_cut[:-1]
    assert neuron.refractory_remaining == 1.0


def test_reset_restores_normalized_source_profile_state() -> None:
    neuron = GLIFNeuron()
    neuron.simulate(100, 30.0, backend="python")

    neuron.reset()

    assert _state(neuron) == (neuron.e_l, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("bad_current", [math.nan, math.inf, -math.inf])
def test_invalid_current_is_failure_atomic(bad_current: float) -> None:
    neuron = GLIFNeuron(v=-68.0, theta_spike=0.5, i_asc1=0.4)
    before = _state(neuron)

    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(bad_current)

    assert _state(neuron) == before


@pytest.mark.parametrize(
    "field",
    ["capacitance", "resistance", "b_spike", "b_voltage", "k_asc1", "k_asc2", "dt"],
)
def test_positive_parameters_are_enforced(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GLIFNeuron(**{field: 0.0})


def test_nonfinite_candidate_is_failure_atomic() -> None:
    neuron = GLIFNeuron(v=1e308, i_asc1=1e308, i_asc2=1e308)
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(1e308)

    assert _state(neuron) == before


def test_public_batch_equals_public_step_sequence() -> None:
    batch = GLIFNeuron()
    trace, events = batch.simulate(512, 30.0, backend="python")
    stepped = GLIFNeuron()
    expected = np.empty(512, dtype=np.float64)
    expected_events = 0
    for index in range(expected.size):
        expected_events += stepped.step(30.0)
        expected[index] = stepped.v

    np.testing.assert_array_equal(trace, expected)
    assert events == expected_events
    assert _state(batch) == _state(stepped)


def test_population_reaches_public_glif5_surface() -> None:
    population = Population(GLIFNeuron, n=3, label="glif5")
    events = [
        neuron.step(current)
        for neuron, current in zip(population.neurons, (30.0, 50.0, 0.0), strict=True)
    ]

    assert all(event in (0, 1) for event in events)
    assert all(math.isfinite(neuron.theta) for neuron in population.neurons)
