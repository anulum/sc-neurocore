# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MihalasNieburNeuron behavioral tests

"""Module-specific behavioral contract for MihalasNieburNeuron.

The contract protects the four-dimensional Mihalas-Niebur continuous flow,
candidate-first spike reset, adaptive threshold coupling, after-spike current
kicks, and fail-closed handling of invalid runtime states.
"""

from __future__ import annotations

from math import inf, isclose, isnan, nan

import pytest

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron


def _state(neuron: MihalasNieburNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.theta, neuron.i1, neuron.i2


def _derivatives(
    neuron: MihalasNieburNeuron,
    v: float,
    theta: float,
    i1: float,
    i2: float,
    current: float,
) -> tuple[float, float, float, float]:
    return (
        (-(v - neuron.v_rest) + i1 + i2 + current) / neuron.tau_v,
        (neuron.theta_inf - theta + neuron.a * (v - neuron.v_rest)) / neuron.tau_theta,
        -i1 / neuron.tau_1,
        -i2 / neuron.tau_2,
    )


def _add_scaled(
    state: tuple[float, float, float, float],
    slope: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    return tuple(value + scale * delta for value, delta in zip(state, slope, strict=True))


def _rk4_reference(
    neuron: MihalasNieburNeuron, current: float
) -> tuple[float, float, float, float]:
    start = _state(neuron)
    half_dt = 0.5 * neuron.dt
    k1 = _derivatives(neuron, *start, current)
    k2 = _derivatives(neuron, *_add_scaled(start, k1, half_dt), current)
    k3 = _derivatives(neuron, *_add_scaled(start, k2, half_dt), current)
    k4 = _derivatives(neuron, *_add_scaled(start, k3, neuron.dt), current)
    slopes = tuple(zip(k1, k2, k3, k4, strict=True))
    return tuple(
        value + neuron.dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        for value, (d1, d2, d3, d4) in zip(start, slopes, strict=True)
    )


def test_subthreshold_step_matches_independent_rk4_reference() -> None:
    neuron = MihalasNieburNeuron(a=0.2, i1=0.3, i2=-0.1)
    expected = _rk4_reference(neuron, current=0.5)

    assert neuron.step(0.5) == 0

    for observed, target in zip(_state(neuron), expected, strict=True):
        assert isclose(observed, target, rel_tol=0.0, abs_tol=1e-15)


def test_after_spike_currents_decay_with_rk4_flow() -> None:
    current = 0.0
    neuron = MihalasNieburNeuron(i1=5.0, i2=5.0)
    expected = _rk4_reference(neuron, current)

    assert neuron.step(current) == 0
    assert isclose(neuron.i1, expected[2], rel_tol=0.0, abs_tol=1e-15)
    assert isclose(neuron.i2, expected[3], rel_tol=0.0, abs_tol=1e-15)
    assert neuron.i1 < neuron.i2


def test_spike_reset_uses_candidate_voltage_and_current_kicks() -> None:
    neuron = MihalasNieburNeuron(v=0.99, b=0.5, r1=1.25, r2=-0.25)
    candidate = _rk4_reference(neuron, current=2.0)

    assert candidate[0] > candidate[1]
    assert neuron.step(2.0) == 1
    assert isclose(neuron.v, neuron.v_reset + neuron.b * (candidate[0] - neuron.v_rest))
    assert isclose(neuron.theta, max(candidate[1], neuron.theta_reset))
    assert isclose(neuron.i1, candidate[2] + neuron.r1)
    assert isclose(neuron.i2, candidate[3] + neuron.r2)


def test_reset_restores_rest_threshold_and_zero_currents() -> None:
    neuron = MihalasNieburNeuron(v=0.9, theta=0.8, i1=1.0, i2=-1.0)
    assert neuron.step(2.0) == 1

    neuron.reset()

    assert _state(neuron) == (neuron.v_rest, neuron.theta_reset, 0.0, 0.0)


@pytest.mark.parametrize("bad_current", [nan, inf, -inf])
def test_invalid_current_preserves_state(bad_current: float) -> None:
    neuron = MihalasNieburNeuron(v=0.2, theta=1.1, i1=0.3, i2=0.4)
    before = _state(neuron)

    assert neuron.step(bad_current) == 0

    assert _state(neuron) == before
    assert isnan(bad_current) or not isclose(bad_current, 0.0)


@pytest.mark.parametrize("field", ["tau_v", "tau_theta", "tau_1", "tau_2", "dt"])
def test_nonpositive_time_constants_preserve_state(field: str) -> None:
    neuron = MihalasNieburNeuron(v=0.2, theta=1.1, i1=0.3, i2=0.4)
    setattr(neuron, field, 0.0)
    before = _state(neuron)

    assert neuron.step(1.0) == 0

    assert _state(neuron) == before


def test_nonfinite_state_preserves_all_fields() -> None:
    neuron = MihalasNieburNeuron(v=nan, theta=1.0, i1=0.0, i2=0.0)
    before = _state(neuron)

    assert neuron.step(1.0) == 0

    assert isnan(neuron.v)
    assert _state(neuron)[1:] == before[1:]


def test_nonfinite_candidate_preserves_state() -> None:
    neuron = MihalasNieburNeuron(v=1e308, theta=1.0, i1=1e308, i2=1e308)
    before = _state(neuron)

    assert neuron.step(1e308) == 0

    assert _state(neuron) == before


def test_dynamic_threshold_couples_to_voltage_in_rk4_flow() -> None:
    low = MihalasNieburNeuron(v=0.25, a=0.0)
    high = MihalasNieburNeuron(v=0.25, a=0.4)

    low.step(0.0)
    high.step(0.0)

    assert high.theta > low.theta
