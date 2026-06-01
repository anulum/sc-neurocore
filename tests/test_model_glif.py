# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIFNeuron behavioural tests

"""Module-specific behavioural contract for GLIFNeuron.

The contract protects the Allen GLIF5 continuous flow, candidate-first RK4
integration, additive threshold reset, after-spike current kicks, validation
boundaries, and public population wiring.
"""

from __future__ import annotations

from math import inf, isclose, isfinite, nan

import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.glif import GLIFNeuron


def _state(neuron: GLIFNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.theta, neuron.i_asc1, neuron.i_asc2


def _derivatives(
    neuron: GLIFNeuron,
    v: float,
    theta: float,
    i_asc1: float,
    i_asc2: float,
    current: float,
) -> tuple[float, float, float, float]:
    return (
        (-(v - neuron.v_rest) + neuron.resistance * current + i_asc1 + i_asc2) / neuron.tau_m,
        (neuron.theta_inf - theta + neuron.a_theta * (v - neuron.v_rest)) / neuron.tau_theta,
        -i_asc1 / neuron.tau_asc1,
        -i_asc2 / neuron.tau_asc2,
    )


def _add_scaled(
    state: tuple[float, float, float, float],
    slope: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    return (
        state[0] + scale * slope[0],
        state[1] + scale * slope[1],
        state[2] + scale * slope[2],
        state[3] + scale * slope[3],
    )


def _rk4_reference(neuron: GLIFNeuron, current: float) -> tuple[float, float, float, float]:
    state = _state(neuron)
    half_dt = 0.5 * neuron.dt
    k1 = _derivatives(neuron, *state, current)
    k2 = _derivatives(neuron, *_add_scaled(state, k1, half_dt), current)
    k3 = _derivatives(neuron, *_add_scaled(state, k2, half_dt), current)
    k4 = _derivatives(neuron, *_add_scaled(state, k3, neuron.dt), current)
    return (
        state[0] + neuron.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + neuron.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + neuron.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        state[3] + neuron.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
    )


def test_subthreshold_step_matches_independent_rk4_reference() -> None:
    neuron = GLIFNeuron(v=-68.0, theta=-45.0, i_asc1=0.4, i_asc2=-0.2)
    expected = _rk4_reference(neuron, current=4.0)

    assert neuron.step(4.0) == 0

    for observed, target in zip(_state(neuron), expected, strict=True):
        assert isclose(observed, target, rel_tol=0.0, abs_tol=1e-14)


def test_after_spike_currents_decay_inside_continuous_flow() -> None:
    neuron = GLIFNeuron(i_asc1=5.0, i_asc2=5.0)
    expected = _rk4_reference(neuron, current=0.0)

    assert neuron.step(0.0) == 0

    assert isclose(neuron.i_asc1, expected[2], rel_tol=0.0, abs_tol=1e-14)
    assert isclose(neuron.i_asc2, expected[3], rel_tol=0.0, abs_tol=1e-14)
    assert neuron.i_asc1 < neuron.i_asc2


def test_spike_reset_is_candidate_first_and_additive() -> None:
    neuron = GLIFNeuron(v=-51.0, theta=-50.5, delta_theta=2.5, r_asc1=1.25, r_asc2=-0.25)
    candidate = _rk4_reference(neuron, current=40.0)

    assert candidate[0] >= candidate[1]
    assert neuron.step(40.0) == 1
    assert neuron.v == neuron.v_reset
    assert isclose(neuron.theta, candidate[1] + neuron.delta_theta, rel_tol=0.0, abs_tol=1e-14)
    assert isclose(neuron.i_asc1, candidate[2] + neuron.r_asc1, rel_tol=0.0, abs_tol=1e-14)
    assert isclose(neuron.i_asc2, candidate[3] + neuron.r_asc2, rel_tol=0.0, abs_tol=1e-14)


def test_reset_restores_rest_threshold_and_zero_currents() -> None:
    neuron = GLIFNeuron(v=-51.0, theta=-50.5, i_asc1=0.5, i_asc2=-0.2)
    assert neuron.step(40.0) == 1

    neuron.reset()

    assert _state(neuron) == (neuron.v_rest, neuron.theta_inf, 0.0, 0.0)


@pytest.mark.parametrize("bad_current", [nan, inf, -inf])
def test_invalid_current_raises_before_state_mutation(bad_current: float) -> None:
    neuron = GLIFNeuron(v=-68.0, theta=-45.0, i_asc1=0.4, i_asc2=-0.2)
    before = _state(neuron)

    with pytest.raises(ValueError, match="current"):
        neuron.step(bad_current)

    assert _state(neuron) == before


@pytest.mark.parametrize("field", ["tau_m", "tau_theta", "tau_asc1", "tau_asc2", "dt"])
def test_nonpositive_time_constants_are_rejected_at_construction(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GLIFNeuron(**{field: 0.0})


@pytest.mark.parametrize("field", ["resistance", "delta_theta"])
def test_negative_nonnegative_parameters_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GLIFNeuron(**{field: -1.0})


def test_corrupted_runtime_state_raises_before_mutation() -> None:
    neuron = GLIFNeuron(v=-68.0, theta=-45.0, i_asc1=0.4, i_asc2=-0.2)
    neuron.theta = nan
    before = _state(neuron)

    with pytest.raises(ValueError, match="theta"):
        neuron.step(1.0)

    assert _state(neuron) == before


def test_nonfinite_candidate_raises_before_mutation() -> None:
    neuron = GLIFNeuron(v=1e308, theta=-45.0, i_asc1=1e308, i_asc2=1e308)
    before = _state(neuron)

    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(1e308)

    assert _state(neuron) == before


def test_voltage_coupling_raises_threshold_relative_to_uncoupled_case() -> None:
    uncoupled = GLIFNeuron(v=-65.0, a_theta=0.0)
    coupled = GLIFNeuron(v=-65.0, a_theta=0.2)

    uncoupled.step(0.0)
    coupled.step(0.0)

    assert coupled.theta > uncoupled.theta


def test_population_wires_public_glif_surface() -> None:
    population = Population(GLIFNeuron, n=3, label="glif")
    spikes = [
        neuron.step(current)
        for neuron, current in zip(population.neurons, [15.0, 15.0, 0.0], strict=True)
    ]

    assert len(spikes) == 3
    assert all(spike in (0, 1) for spike in spikes)
    assert all(isfinite(neuron.v) for neuron in population.neurons)
