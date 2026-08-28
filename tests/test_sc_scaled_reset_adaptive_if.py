# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained scaled-reset adaptive IF behavioural tests

"""Independent recurrence and public-surface tests for the retained SC identity."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)


def _state(neuron: SCScaledResetAdaptiveIFNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.theta, neuron.i1, neuron.i2


def _derivatives(
    neuron: SCScaledResetAdaptiveIFNeuron,
    state: tuple[float, float, float, float],
    current: float,
) -> tuple[float, float, float, float]:
    v, theta, i1, i2 = state
    return (
        (-(v - neuron.v_rest) + i1 + i2 + current) / neuron.tau_v,
        (neuron.theta_inf - theta + neuron.a * (v - neuron.v_rest)) / neuron.tau_theta,
        -i1 / neuron.tau_1,
        -i2 / neuron.tau_2,
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


def _candidate(
    neuron: SCScaledResetAdaptiveIFNeuron, current: float
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


def test_subthreshold_flow_matches_retained_rk4() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=0.2, theta=1.2, i1=0.3, i2=-0.1, a=0.2)
    expected = _candidate(neuron, 0.5)

    assert neuron.step(0.5) == 0

    np.testing.assert_allclose(_state(neuron), expected, rtol=0.0, atol=1e-15)


def test_scaled_voltage_reset_and_additive_kicks_are_preserved() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=0.99, b=0.5, r1=1.25, r2=-0.25)
    candidate = _candidate(neuron, 2.0)

    assert candidate[0] >= candidate[1]
    assert neuron.step(2.0) == 1
    assert neuron.v == neuron.v_reset + neuron.b * (candidate[0] - neuron.v_rest)
    assert neuron.theta == max(candidate[1], neuron.theta_reset)
    assert neuron.i1 == candidate[2] + neuron.r1
    assert neuron.i2 == candidate[3] + neuron.r2


def test_historical_default_regimes_are_pinned() -> None:
    for current, expected in ((0.0, 0), (2.0, 142), (5.0, 333)):
        neuron = SCScaledResetAdaptiveIFNeuron()
        _, events = neuron.simulate(1000, current, backend="python")
        assert events == expected


@pytest.mark.parametrize("bad_current", [math.nan, math.inf, -math.inf])
def test_invalid_current_is_failure_atomic(bad_current: float) -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=0.2, theta=1.2, i1=0.3)
    before = _state(neuron)

    with pytest.raises(ValueError, match="current"):
        neuron.step(bad_current)

    assert _state(neuron) == before


@pytest.mark.parametrize("field", ["tau_v", "tau_theta", "tau_1", "tau_2", "dt"])
def test_positive_time_constants_are_enforced(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SCScaledResetAdaptiveIFNeuron(**{field: 0.0})


def test_nonfinite_candidate_is_failure_atomic() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=1e308, theta=1e308, i1=1e308, i2=1e308)
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(1e308)

    assert _state(neuron) == before


@pytest.mark.parametrize("field", ["v", "theta", "a", "r2"])
def test_nonfinite_state_and_reset_parameters_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SCScaledResetAdaptiveIFNeuron(**{field: math.nan})


def test_nonfinite_reset_state_is_failure_atomic() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(
        v=0.99,
        i1=1e308,
        tau_1=1e308,
        r1=1e308,
    )
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="state"):
        neuron.step(2.0)

    assert _state(neuron) == before


def test_public_batch_matches_public_step_sequence() -> None:
    parameters = dict(
        theta_reset=1.3, tau_theta=40.0, tau_1=15.0, tau_2=80.0, a=0.1, b=0.1, r1=0.2, r2=-0.15
    )
    drive = (0.0, 3.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.5) * 200
    batch = SCScaledResetAdaptiveIFNeuron(**parameters)
    trace, events = batch.simulate(len(drive), 3.0, backend="python")
    constant = SCScaledResetAdaptiveIFNeuron(**parameters)
    expected = np.empty(len(drive), dtype=np.float64)
    expected_events = 0
    for index in range(len(drive)):
        expected_events += constant.step(3.0)
        expected[index] = constant.v

    np.testing.assert_array_equal(trace, expected)
    assert events == expected_events
    assert _state(batch) == _state(constant)


def test_population_reaches_retained_surface() -> None:
    population = Population(SCScaledResetAdaptiveIFNeuron, n=3, label="scaled-reset")
    events = [
        neuron.step(current)
        for neuron, current in zip(population.neurons, (0.0, 2.0, 5.0), strict=True)
    ]

    assert all(event in (0, 1) for event in events)
    assert all(math.isfinite(neuron.theta) for neuron in population.neurons)


def test_reset_restores_retained_stationary_state() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron(v=2.0, theta=3.0, i1=4.0, i2=-5.0)

    neuron.reset()

    assert _state(neuron) == (neuron.v_rest, neuron.theta_reset, 0.0, 0.0)
