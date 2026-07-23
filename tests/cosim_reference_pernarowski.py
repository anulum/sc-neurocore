# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski co-simulation references

"""Independent Pernarowski spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from tests.cosim_reference_statistics import _summarise


def _pernarowski_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Pernarowski RK4 upward-crossing count."""
    neuron = PernarowskiNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _pernarowski_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the autonomous Pernarowski flow.

    The Pernarowski (1994) beta-cell model couples a fast cubic coordinate to
    recovery ``w`` and ultra-slow adaptation ``z``. This independent recurrence
    advances all three equations simultaneously with classical four-stage RK4,
    then applies the maintained rising-edge ``v >= 0.5`` crossing decision
    without resetting state. It is re-derived here rather than calling the hand
    model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v``, ``w``, and ``z``, plus the spike count
        and first-spike step.
    """
    alpha = 0.1
    beta = 0.5
    eps1 = 0.1
    eps2 = 0.001
    gamma = 0.5
    threshold = 0.5
    v = -1.0
    w = 0.0
    z = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, z_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state - z_state + current,
            eps1 * (v_state - gamma * w_state + alpha),
            eps2 * (beta * (v_state + 0.7) - z_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, z)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], z + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        z = z + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        z_values.append(z)

    return _summarise({"v": v_values, "w": w_values, "z": z_values}, spikes)
