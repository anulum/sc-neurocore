# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR co-simulation references

"""Independent Wilson-HR spike-count and RK4 reference contracts."""

from __future__ import annotations

from tests.cosim_reference_statistics import _summarise


def _wilson_hr_hand_spike_count(n_steps: int, current: float) -> int:
    """Return an independently re-derived Wilson-HR upward-crossing count."""
    features = _wilson_hr_rk4_features(current=current, dt=0.05, steps=n_steps)
    return int(features["spike_count"])


def _wilson_hr_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Wilson-HR cortical model.

    This independent recurrence re-derives Wilson's two-state polynomial flow,
    advances ``v`` and ``r`` simultaneously through four Runge-Kutta stages, and
    applies an observational upward crossing at ``v = 0`` without resetting the
    continuous source flow. The helper does not call the hand model or schema runner.

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
        Reference features for post-reset ``v`` and candidate ``r``, plus the
        spike count and first-spike step.
    """
    tau_r = 1.9
    capacitance = 0.8
    threshold = 0.0
    v = -0.7
    r = 0.085
    v_values: list[float] = []
    r_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, r_state: float) -> tuple[float, float]:
        membrane = -(17.81 + 47.71 * v_state + 32.63 * v_state * v_state) * (v_state - 0.55)
        recovery_coupling = -26.0 * r_state * (v_state + 0.92)
        return (
            (membrane + recovery_coupling + current) / capacitance,
            (-r_state + 1.35 * v_state + 1.03) / tau_r,
        )

    for _ in range(steps):
        previous_v = v
        k1 = deriv(v, r)
        k2 = deriv(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], r + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        r = r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spike = int(v >= threshold and previous_v < threshold)
        spikes.append(spike)
        v_values.append(v)
        r_values.append(r)

    return _summarise({"v": v_values, "r": r_values}, spikes)


def _sc_resetting_wilson_hr_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return an independent feature receipt for the retained project recurrence."""
    tau_r = 1.9
    threshold = 0.4
    v = -0.7
    r = 0.1
    v_values: list[float] = []
    r_values: list[float] = []
    events: list[int] = []

    def derivatives(voltage: float, recovery: float) -> tuple[float, float]:
        polynomial = -(17.81 + 47.71 * voltage + 32.63 * voltage * voltage) * (voltage - 0.55)
        recovery_current = -26.0 * recovery * (voltage + 0.92)
        return (
            polynomial + recovery_current + current,
            (-recovery + 1.35 * voltage + 1.03) / tau_r,
        )

    for _ in range(steps):
        k1 = derivatives(v, r)
        k2 = derivatives(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1])
        k3 = derivatives(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1])
        k4 = derivatives(v + dt * k3[0], r + dt * k3[1])
        next_v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        r = r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        event = int(next_v >= threshold)
        v = -0.7 if event else next_v
        events.append(event)
        v_values.append(v)
        r_values.append(r)

    return _summarise({"v": v_values, "r": r_values}, events)
