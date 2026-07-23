# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF co-simulation references

"""Independent GLIF spike-count and driven-RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.glif import GLIFNeuron
from tests.cosim_reference_statistics import _summarise


def _glif_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored GLIF candidate-first RK4 spike count."""
    neuron = GLIFNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _glif_driven_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven GLIF5 flow and adaptive reset.

    The maintained Allen Institute GLIF5 model advances four coupled linear states:
    the membrane potential, adaptive threshold, and two after-spike currents. This
    independent recurrence evaluates all four classical RK4 stages from the same
    pre-step state, then applies the candidate-level ``v >= theta`` decision and the
    candidate-first voltage, threshold, and current reset increments. A driven tonic
    train therefore exercises both the continuous flow and every reset surface rather
    than validating only a silent linear tail.

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
        Reference feature map for the ``v``, ``theta``, ``i_asc1``, and ``i_asc2``
        state variables plus spike-count and first-spike-step features.
    """
    theta_inf = -50.0
    v_rest = -70.0
    v_reset = -70.0
    tau_m = 10.0
    tau_theta = 100.0
    tau_asc1 = 10.0
    tau_asc2 = 200.0
    a_theta = 0.01
    delta_theta = 2.0
    r_asc1 = 1.0
    r_asc2 = 0.5
    resistance = 1.0

    v = v_rest
    theta = theta_inf
    i_asc1 = 0.0
    i_asc2 = 0.0
    half_dt = 0.5 * dt
    recorded: dict[str, list[float]] = {"v": [], "theta": [], "i_asc1": [], "i_asc2": []}
    spikes: list[int] = []

    def derivatives(
        membrane: float,
        threshold: float,
        asc1: float,
        asc2: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(membrane - v_rest) + resistance * current + asc1 + asc2) / tau_m,
            (theta_inf - threshold + a_theta * (membrane - v_rest)) / tau_theta,
            -asc1 / tau_asc1,
            -asc2 / tau_asc2,
        )

    for _ in range(steps):
        k1 = derivatives(v, theta, i_asc1, i_asc2)
        k2 = derivatives(
            v + half_dt * k1[0],
            theta + half_dt * k1[1],
            i_asc1 + half_dt * k1[2],
            i_asc2 + half_dt * k1[3],
        )
        k3 = derivatives(
            v + half_dt * k2[0],
            theta + half_dt * k2[1],
            i_asc1 + half_dt * k2[2],
            i_asc2 + half_dt * k2[3],
        )
        k4 = derivatives(
            v + dt * k3[0],
            theta + dt * k3[1],
            i_asc1 + dt * k3[2],
            i_asc2 + dt * k3[3],
        )
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        i_asc1 = i_asc1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i_asc2 = i_asc2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if v >= theta:
            spikes.append(1)
            v = v_reset
            theta = theta + delta_theta
            i_asc1 = i_asc1 + r_asc1
            i_asc2 = i_asc2 + r_asc2
        else:
            spikes.append(0)
        recorded["v"].append(v)
        recorded["theta"].append(theta)
        recorded["i_asc1"].append(i_asc1)
        recorded["i_asc2"].append(i_asc2)

    return _summarise(recorded, spikes)
