# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Rinzel co-simulation references

"""Independent FitzHugh-Rinzel spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from tests.cosim_reference_statistics import _summarise


def _fitzhugh_rinzel_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored FitzHugh-Rinzel RK4 upward-crossing count."""
    neuron = FitzHughRinzelNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _fitzhugh_rinzel_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven FitzHugh-Rinzel flow.

    The Rinzel (1987) three-state qualitative burster extends the FitzHugh-Nagumo
    fast subsystem with the ultra-slow ``y`` modulation equation. This independent
    recurrence advances all three coupled equations with one simultaneous four-stage
    RK4 step, then applies the maintained rising-edge ``v >= 1`` crossing decision
    without resetting any state. The cube is written ``v * v * v`` to reproduce the
    exact IEEE operation order of the hand model and schema runner; the recurrence is
    re-derived here rather than calling either implementation.

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
        Reference features for ``v``, ``w``, and ``y``, plus the spike count and
        first-spike step.
    """
    a = 0.7
    b = 0.8
    c = -0.775
    d = 1.0
    delta = 0.08
    mu = 0.0001
    threshold = 1.0
    v = -1.0
    w = -0.5
    y = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, y_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + y_state + current,
            delta * (a + v_state - b * w_state),
            mu * (c - v_state - d * y_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, y)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            y + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            y + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], y + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        y = y + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        y_values.append(y)

    return _summarise({"v": v_values, "w": w_values, "y": y_values}, spikes)
