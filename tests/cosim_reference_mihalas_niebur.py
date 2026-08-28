# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Mihalas-Niebur and retained-recurrence references

"""Hand-derived RK4 feature oracles for both Model 14 identities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from tests.cosim_reference_statistics import _summarise

State: TypeAlias = tuple[float, float, float, float]
Derivative: TypeAlias = Callable[[State], State]

_MIHALAS_NIEBUR_PARAMS = {
    "current_jump_1": 0.01,
    "current_jump_2": -0.0006,
}


def _rk4(
    state: State,
    dt: float,
    derivative: Derivative,
) -> State:
    def add(
        values: State,
        slope: State,
        scale: float,
    ) -> State:
        return cast(
            State,
            tuple(value + scale * delta for value, delta in zip(values, slope, strict=True)),
        )

    k1 = derivative(state)
    k2 = derivative(add(state, k1, 0.5 * dt))
    k3 = derivative(add(state, k2, 0.5 * dt))
    k4 = derivative(add(state, k3, dt))
    return cast(
        State,
        tuple(
            value + dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
            for value, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4, strict=True)
        ),
    )


def _features(states: list[State], events: list[int]) -> dict[str, float]:
    traces = [list(values) for values in zip(*states, strict=True)]
    first = events.index(1) if 1 in events else -2
    return _summarise(
        {"v": traces[0], "theta": traces[1], "i1": traces[2], "i2": traces[3]},
        events,
    ) | {"first_spike_step": float(first + 1)}


def _mihalas_niebur_hand_spike_count(n_steps: int, current: float) -> int:
    """Return source-model events through the public one-step API."""
    neuron = MihalasNieburNeuron(**_MIHALAS_NIEBUR_PARAMS)
    return sum(neuron.step(current) for _ in range(n_steps))


def _mihalas_niebur_driven_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Re-derive equations 2.1–2.2 without importing production helpers."""
    state = (-0.07, -0.05, 0.0, 0.0)
    states: list[State] = []
    events: list[int] = []

    def derivative(values: State) -> State:
        v, theta, i1, i2 = values
        return (
            current + i1 + i2 - 0.05 * (v + 0.07),
            0.005 * (v + 0.07) - 0.01 * (theta + 0.05),
            -0.2 * i1,
            -0.02 * i2,
        )

    for _ in range(steps):
        v, theta, i1, i2 = _rk4(state, dt, derivative)
        event = int(v >= theta)
        if event:
            i1 = 0.01
            i2 -= 0.0006
            v = -0.07
            theta = max(-0.06, theta)
        state = (v, theta, i1, i2)
        states.append(state)
        events.append(event)
    return _features(states, events)


def _sc_scaled_reset_driven_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Re-derive the retained project recurrence at its schema profile."""
    state = (0.0, 1.0, 0.0, 0.0)
    states: list[State] = []
    events: list[int] = []

    def derivative(values: State) -> State:
        v, theta, i1, i2 = values
        return (
            (-v + i1 + i2 + current) / 10.0,
            (1.0 - theta + 0.1 * v) / 40.0,
            -i1 / 15.0,
            -i2 / 80.0,
        )

    for _ in range(steps):
        v, theta, i1, i2 = _rk4(state, dt, derivative)
        event = int(v >= theta)
        if event:
            v = 0.1 * v
            theta = max(theta, 1.3)
            i1 += 0.2
            i2 -= 0.15
        state = (v, theta, i1, i2)
        states.append(state)
        events.append(event)
    return _features(states, events)
