# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose co-simulation references

"""Independent Hindmarsh-Rose spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from tests.cosim_reference_statistics import _summarise


def _hindmarsh_rose_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Hindmarsh-Rose RK4 upward-crossing count."""
    neuron = HindmarshRoseNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _hindmarsh_rose_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven Hindmarsh-Rose flow.

    The Hindmarsh-Rose (1984) cubic fast subsystem and slow adaptation variable are
    advanced with an independently re-derived simultaneous four-stage RK4 step. The
    maintained event is an upward ``x >= 1`` crossing and does not reset any state.
    Repeated multiplication preserves the source polynomial's evaluation order without
    importing either the hand model or the schema runner.

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
        Reference feature map for the ``x``, ``y``, and ``z`` state variables plus
        spike-count and first-spike-step features.
    """
    b = 3.0
    r = 0.001
    s = 4.0
    x_rest = -1.6
    threshold = 1.0
    x = -1.6
    y = -10.0
    z = 2.0
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []

    def derivatives(x_state: float, y_state: float, z_state: float) -> tuple[float, float, float]:
        x2 = x_state * x_state
        x3 = x2 * x_state
        return (
            y_state - x3 + b * x2 - z_state + current,
            1.0 - 5.0 * x2 - y_state,
            r * (s * (x_state - x_rest) - z_state),
        )

    for _ in range(steps):
        x_prev = x
        k1 = derivatives(x, y, z)
        k2 = derivatives(
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
        )
        k3 = derivatives(
            x + 0.5 * dt * k2[0],
            y + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
        )
        k4 = derivatives(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x = x + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y = y + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        z = z + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        spikes.append(1 if (x >= threshold and x_prev < threshold) else 0)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return _summarise({"x": x_values, "y": y_values, "z": z_values}, spikes)
