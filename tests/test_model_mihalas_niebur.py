# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-faithful Mihalas-Niebur behavioural tests

"""Independent equation and public-surface contracts for MihalasNieburNeuron."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron


def _state(neuron: MihalasNieburNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.theta, neuron.i1, neuron.i2


def _derivatives(
    neuron: MihalasNieburNeuron,
    state: tuple[float, float, float, float],
    current: float,
) -> tuple[float, float, float, float]:
    v, theta, i1, i2 = state
    return (
        current + i1 + i2 - neuron.leak_rate * (v - neuron.v_rest),
        neuron.threshold_voltage_coupling * (v - neuron.v_rest)
        - neuron.threshold_decay_rate * (theta - neuron.theta_inf),
        -neuron.current_decay_rate_1 * i1,
        -neuron.current_decay_rate_2 * i2,
    )


def _add(
    state: tuple[float, float, float, float],
    slope: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    return cast(
        tuple[float, float, float, float],
        tuple(value + scale * delta for value, delta in zip(state, slope, strict=True)),
    )


def _source_candidate(
    neuron: MihalasNieburNeuron, current: float
) -> tuple[float, float, float, float]:
    state = _state(neuron)
    half_dt = 0.5 * neuron.dt
    k1 = _derivatives(neuron, state, current)
    k2 = _derivatives(neuron, _add(state, k1, half_dt), current)
    k3 = _derivatives(neuron, _add(state, k2, half_dt), current)
    k4 = _derivatives(neuron, _add(state, k3, neuron.dt), current)
    return cast(
        tuple[float, float, float, float],
        tuple(
            value + neuron.dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
            for value, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4, strict=True)
        ),
    )


def test_equation_2_1_flow_matches_independent_rk4() -> None:
    neuron = MihalasNieburNeuron(v=-0.065, theta=-0.04, i1=0.001, i2=-0.0002)
    expected = _source_candidate(neuron, 0.0005)

    assert neuron.step(0.0005) == 0

    np.testing.assert_allclose(_state(neuron), expected, rtol=0.0, atol=1e-17)


def test_equation_2_2_event_map_is_applied_to_candidate_currents() -> None:
    neuron = MihalasNieburNeuron(
        v=-0.0501,
        theta=-0.05,
        i1=0.003,
        i2=-0.001,
        current_retention_1=0.25,
        current_retention_2=0.5,
        current_jump_1=0.004,
        current_jump_2=0.002,
    )
    candidate = _source_candidate(neuron, 0.02)

    assert candidate[0] >= candidate[1]
    assert neuron.step(0.02) == 1
    assert neuron.v == neuron.v_reset
    assert neuron.theta == max(neuron.theta_reset, candidate[1])
    assert neuron.i1 == 0.25 * candidate[2] + 0.004
    assert neuron.i2 == 0.5 * candidate[3] + 0.002


def test_table_1_panel_m_profile_forms_two_bursts() -> None:
    neuron = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    event_indices = [index for index in range(2000) if neuron.step(0.002)]

    assert event_indices == [
        146,
        170,
        197,
        226,
        259,
        296,
        339,
        391,
        1433,
        1468,
        1507,
        1551,
        1602,
        1665,
    ]


def test_reset_restores_stationary_source_state() -> None:
    neuron = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    neuron.simulate(500, 0.002, backend="python")

    neuron.reset()

    assert _state(neuron) == (neuron.v_rest, neuron.theta_inf, 0.0, 0.0)


@pytest.mark.parametrize("bad_current", [math.nan, math.inf, -math.inf])
def test_invalid_current_is_failure_atomic(bad_current: float) -> None:
    neuron = MihalasNieburNeuron(v=-0.065, i1=0.001)
    before = _state(neuron)

    with pytest.raises(ValueError, match="current"):
        neuron.step(bad_current)

    assert _state(neuron) == before


@pytest.mark.parametrize(
    "field",
    [
        "leak_rate",
        "threshold_decay_rate",
        "current_decay_rate_1",
        "current_decay_rate_2",
        "dt",
    ],
)
def test_positive_rates_are_enforced(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        MihalasNieburNeuron(**{field: 0.0})


def test_source_reset_constraint_is_enforced() -> None:
    with pytest.raises(ValueError, match="theta_reset"):
        MihalasNieburNeuron(v_reset=-0.05, theta_reset=-0.05)


def test_nonfinite_candidate_is_failure_atomic() -> None:
    neuron = MihalasNieburNeuron(v=1e308, theta=1e308, i1=1e308, i2=1e308)
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(1e308)

    assert _state(neuron) == before


@pytest.mark.parametrize("field", ["v", "theta", "current_jump_1", "current_retention_2"])
def test_nonfinite_state_and_event_parameters_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        MihalasNieburNeuron(**{field: math.nan})


def test_nonfinite_event_map_is_failure_atomic() -> None:
    neuron = MihalasNieburNeuron(
        v=-0.0501,
        theta=-0.05,
        i1=2.0,
        current_retention_1=1e308,
    )
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="reset state"):
        neuron.step(0.02)

    assert _state(neuron) == before


def test_public_batch_matches_public_step_sequence() -> None:
    batch = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    trace, events = batch.simulate(2000, 0.002, backend="python")
    stepped = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    expected = np.empty(2000, dtype=np.float64)
    expected_events = 0
    for index in range(expected.size):
        expected_events += stepped.step(0.002)
        expected[index] = stepped.v

    np.testing.assert_array_equal(trace, expected)
    assert events == expected_events == 14
    assert _state(batch) == _state(stepped)


def test_population_reaches_source_model_surface() -> None:
    population = Population(MihalasNieburNeuron, n=3, label="mihalas-niebur")
    events = [
        neuron.step(current)
        for neuron, current in zip(population.neurons, (0.0, 0.002, -0.001), strict=True)
    ]

    assert all(event in (0, 1) for event in events)
    assert all(math.isfinite(neuron.theta) for neuron in population.neurons)
