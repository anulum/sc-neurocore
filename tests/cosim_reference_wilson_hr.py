# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR co-simulation references

"""Independent Wilson-HR spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from tests.cosim_reference_statistics import _summarise


def _wilson_hr_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Wilson-HR RK4 hard-reset spike count."""
    neuron = WilsonHRNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _wilson_hr_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Wilson-HR cortical model.

    This independent recurrence re-derives Wilson's two-state polynomial flow,
    advances ``v`` and ``r`` simultaneously through four Runge-Kutta stages, and
    applies the level ``v >= 0.4`` spike decision. A spike hard-resets only ``v``
    to ``-0.7``; the RK4 candidate recovery state is preserved. The helper does not
    call the hand model or schema runner.

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
    threshold = 0.4
    reset_voltage = -0.7
    v = -0.7
    r = 0.1
    v_values: list[float] = []
    r_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, r_state: float) -> tuple[float, float]:
        membrane = -(17.81 + 47.71 * v_state + 32.63 * v_state * v_state) * (v_state - 0.55)
        recovery_coupling = -26.0 * r_state * (v_state + 0.92)
        return (
            membrane + recovery_coupling + current,
            (-r_state + 1.35 * v_state + 1.03) / tau_r,
        )

    for _ in range(steps):
        k1 = deriv(v, r)
        k2 = deriv(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], r + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        r = r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spike = int(v >= threshold)
        if spike:
            v = reset_voltage
        spikes.append(spike)
        v_values.append(v)
        r_values.append(r)

    return _summarise({"v": v_values, "r": r_values}, spikes)
