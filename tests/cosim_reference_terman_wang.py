# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman-Wang co-simulation references

"""Independent Terman-Wang spike-count and RK4 reference contracts."""

from __future__ import annotations

import math

from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from tests.cosim_reference_statistics import _summarise


def _terman_wang_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Terman-Wang RK4 upward-crossing count."""
    neuron = TermanWangOscillator()
    return sum(neuron.step(current) for _ in range(n_steps))


def _terman_wang_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Terman-Wang LEGION oscillator.

    This independent recurrence re-derives the maintained two-state Terman-Wang
    (1995) cubic fast nullcline and ``tanh``-gated slow recovery equation. It
    advances both states simultaneously through four Runge-Kutta stages, then
    applies the no-reset rising-edge ``v >= 1.5`` crossing decision without
    calling the hand model or schema runner.

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
        Reference features for ``v`` and ``w``, plus the spike count and
        first-spike step.
    """
    alpha = 3.0
    beta = 0.2
    epsilon = 0.02
    rho = 0.0
    threshold = 1.5
    v = -1.5
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        fast = 3.0 * v_state - v_state * v_state * v_state + 2.0
        recovery = alpha * (1.0 + math.tanh(v_state / beta))
        return fast - w_state + current + rho, epsilon * (recovery - w_state)

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w)
        k2 = deriv(v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], w + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)
