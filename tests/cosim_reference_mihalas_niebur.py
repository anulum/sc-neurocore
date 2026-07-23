# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur co-simulation references

"""Independent Mihalas-Niebur spike-count and driven-RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from tests.cosim_reference_statistics import _summarise

# Adaptive-threshold operating point mirrored by the bundled ``mihalas_niebur`` schema.
# ``theta_reset`` (1.3) exceeds ``theta_inf`` (1.0), so the max() threshold floor engages
# on every spike and the fractional taus/coefficients stress the fixed-point datapath.
_MIHALAS_NIEBUR_PARAMS = {
    "v_rest": 0.0,
    "v_reset": 0.0,
    "theta_reset": 1.3,
    "theta_inf": 1.0,
    "tau_v": 10.0,
    "tau_theta": 40.0,
    "tau_1": 15.0,
    "tau_2": 80.0,
    "a": 0.1,
    "b": 0.1,
    "r1": 0.2,
    "r2": -0.15,
}


def _mihalas_niebur_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Mihalas-Niebur (RK4) spike count for comparison."""
    neuron = MihalasNieburNeuron(dt=1.0, **_MIHALAS_NIEBUR_PARAMS)
    return sum(neuron.step(current) for _ in range(n_steps))


def _mihalas_niebur_driven_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact fourth-order Runge-Kutta features for the driven Mihalas-Niebur flow.

    The generalised integrate-and-fire flow (Mihalaş & Niebur 2009) advances four linear
    states — membrane ``dv/dt = (-(v - v_rest) + i1 + i2 + I) / tau_v``, adaptive threshold
    ``dtheta/dt = (theta_inf - theta + a (v - v_rest)) / tau_theta`` and two spike-triggered
    currents ``di1/dt = -i1 / tau_1``, ``di2/dt = -i2 / tau_2`` — with the classical RK4
    step the schema runner applies, and the adaptive reset ``v = v_reset + b (v - v_rest)``,
    ``theta = max(theta, theta_reset)``, ``i1 += r1``, ``i2 += r2`` fires whenever the
    post-step membrane reaches the state-to-state ``v >= theta`` threshold. Every derivative
    is linear, so the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed driven-spiking trace, not a copy of the runner. Because
    ``theta_reset`` (1.3) exceeds ``theta_inf`` (1.0) the max() threshold floor engages on
    every spike, so the state-to-state comparison is a genuine adaptive threshold.

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
        Reference feature map for the ``v``, ``theta``, ``i1`` and ``i2`` state variables
        plus spike-count and first-spike-step features.
    """
    v_rest = 0.0
    v_reset = 0.0
    theta_reset = 1.3
    theta_inf = 1.0
    tau_v = 10.0
    tau_theta = 40.0
    tau_1 = 15.0
    tau_2 = 80.0
    a = 0.1
    b = 0.1
    r1 = 0.2
    r2 = -0.15
    v = 0.0
    theta = 1.0
    i1 = 0.0
    i2 = 0.0
    half_dt = 0.5 * dt
    v_values: list[float] = []
    theta_values: list[float] = []
    i1_values: list[float] = []
    i2_values: list[float] = []
    spikes: list[int] = []

    def deriv(vv: float, th: float, j1: float, j2: float) -> tuple[float, float, float, float]:
        return (
            (-(vv - v_rest) + j1 + j2 + current) / tau_v,
            (theta_inf - th + a * (vv - v_rest)) / tau_theta,
            -j1 / tau_1,
            -j2 / tau_2,
        )

    for _ in range(steps):
        k1 = deriv(v, theta, i1, i2)
        k2 = deriv(
            v + half_dt * k1[0],
            theta + half_dt * k1[1],
            i1 + half_dt * k1[2],
            i2 + half_dt * k1[3],
        )
        k3 = deriv(
            v + half_dt * k2[0],
            theta + half_dt * k2[1],
            i1 + half_dt * k2[2],
            i2 + half_dt * k2[3],
        )
        k4 = deriv(
            v + dt * k3[0],
            theta + dt * k3[1],
            i1 + dt * k3[2],
            i2 + dt * k3[3],
        )
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        i1 = i1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i2 = i2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if v >= theta:
            spikes.append(1)
            v = v_reset + b * (v - v_rest)
            theta = max(theta, theta_reset)
            i1 = i1 + r1
            i2 = i2 + r2
        else:
            spikes.append(0)
        v_values.append(v)
        theta_values.append(theta)
        i1_values.append(i1)
        i2_values.append(i2)

    return _summarise(
        {"v": v_values, "theta": theta_values, "i1": i1_values, "i2": i2_values}, spikes
    )
